"""
Download CME FX futures (6E = Euro FX) L3 (MBO) data from Databento and turn
it into structured, book-annotated parquet.

Setup:
    pip install -U databento pandas pyarrow sortedcontainers
    set DATABENTO_API_KEY (user env var)        # never hard-code the key

Usage (always run in this order):
    python main.py defs 2026-09-14
    python main.py cost 6EZ6 2026-09-14 2026-10-14 mbo
    python main.py pull-days 6EZ6 2026-09-14 2026-10-14 --max-usd 6   # one file per UTC day
    python main.py build 6EZ6                                          # raw/ -> l3/
    python main.py snap 6EZ6 2026-09-15T13:30                          # order-level book at a time

Dates are UTC and `end` is exclusive. The MBO book can only be rebuilt from the
00:00 UTC snapshot at the start of each day, so pulls are whole UTC days.
"""
import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sortedcontainers import SortedDict

DATASET = "GLBX.MDP3"          # CME Globex MDP 3.0
OUT = Path("data_6e")
RAW = OUT / "raw"              # Databento DBN files, one per UTC day
L3 = OUT / "l3"                # structured parquet, one per UTC day
for d in (OUT, RAW, L3):
    d.mkdir(exist_ok=True)

UNDEF_PRICE = 2**63 - 1        # Databento's null price
PX_SCALE = 1e-9                # DBN prices are fixed-point int64 * 1e-9


def client() -> db.Historical:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("Set DATABENTO_API_KEY in your environment.")
    return db.Historical(key)


# ---------------------------------------------------------------- download

def cmd_defs(c, day):
    """All 6E contracts alive on `day`: raw symbol, tick size, matching algo, expiry."""
    data = c.timeseries.get_range(
        dataset=DATASET, stype_in="parent", symbols=["6E.FUT"],
        schema="definition", start=day,
    )
    df = data.to_df()
    cols = ["instrument_id", "raw_symbol", "min_price_increment",
            "match_algorithm", "expiration"]
    df = df[cols].drop_duplicates("instrument_id").sort_values("expiration")
    print(df.to_string(index=False))
    df.to_csv(OUT / "6e_definitions.csv", index=False)


def cmd_cost(c, sym, start, end, schema):
    kw = dict(dataset=DATASET, symbols=[sym], stype_in="raw_symbol",
              schema=schema, start=start, end=end)
    cost = c.metadata.get_cost(**kw)
    size = c.metadata.get_billable_size(**kw)
    print(f"{schema} {sym} {start}->{end}: ${cost:.2f}, {size/1e9:.2f} GB")
    return cost


def cmd_pull(c, sym, start, end, schema, max_usd):
    cost = cmd_cost(c, sym, start, end, schema)
    if cost > max_usd:
        sys.exit(f"Refusing: ${cost:.2f} > --max-usd {max_usd}")
    tag = f"{start}_{end}".replace(":", "")      # ':' is illegal in Windows filenames
    path = OUT / f"{sym}_{schema}_{tag}.dbn.zst"
    c.timeseries.get_range(
        dataset=DATASET, symbols=[sym], stype_in="raw_symbol",
        schema=schema, start=start, end=end, path=str(path),
    )
    db.DBNStore.from_file(path).to_parquet(str(path).replace(".dbn.zst", ".parquet"))
    print("saved", path)


def cmd_pull_days(c, sym, start, end, max_usd, schema="mbo"):
    """One DBN file per UTC day; files already on disk are skipped (resumable)."""
    cost = cmd_cost(c, sym, start, end, schema)
    if cost > max_usd:
        sys.exit(f"Refusing: ${cost:.2f} > --max-usd {max_usd}")
    d, stop = date.fromisoformat(start), date.fromisoformat(end)
    while d < stop:
        path = RAW / f"{sym}_{schema}_{d}.dbn.zst"
        if path.exists():
            print("have", path.name)
        else:
            tmp = path.with_suffix(".part")
            tmp.unlink(missing_ok=True)
            c.timeseries.get_range(
                dataset=DATASET, symbols=[sym], stype_in="raw_symbol", schema=schema,
                start=d.isoformat(), end=(d + timedelta(days=1)).isoformat(), path=str(tmp),
            )
            tmp.rename(path)
            print(f"saved {path.name} ({path.stat().st_size/1e6:.1f} MB)")
        d += timedelta(days=1)


# ---------------------------------------------------------------- L3 book

class Book:
    """Order-level (L3) book for one instrument, FIFO queue per price level.

    Databento MBO semantics (checked against GLBX data):
      A add | C cancel (removes `size`; full cancel follows a complete fill)
      M modify (new price/size; price change or size increase loses priority,
        size decrease -- e.g. after a partial fill -- keeps it)
      R clear book | T trade, F fill, N none: no book change
    """

    def __init__(self):
        self.orders = {}                                  # oid -> (side, px, size)
        self.levels = {"B": SortedDict(), "A": SortedDict()}  # px -> {oid: size}, FIFO

    def clear(self):
        self.orders.clear()
        self.levels["B"].clear()
        self.levels["A"].clear()

    def add(self, oid, side, px, sz):
        self.orders[oid] = (side, px, sz)
        self.levels[side].setdefault(px, {})[oid] = sz # FIFO

    def remove(self, oid):
        side, px, _ = self.orders.pop(oid)
        q = self.levels[side][px]
        del q[oid]
        if not q:
            del self.levels[side][px]

    def apply(self, action, oid, side, px, sz):
        """Returns False if the event referenced an order we don't know."""
        if action == "A":
            self.add(oid, side, px, sz)
        elif action == "C":
            if oid not in self.orders:
                return False
            s, p, rest = self.orders[oid]
            if rest - sz > 0:
                self.orders[oid] = (s, p, rest - sz)
                self.levels[s][p][oid] = rest - sz
            else:
                self.remove(oid)
        elif action == "M":
            if oid not in self.orders:                    # treat unknown modify as add
                self.add(oid, side, px, sz)
                return False
            s, p, rest = self.orders[oid]
            if px != p or sz > rest or side != s:
                self.remove(oid)
                self.add(oid, side, px, sz)
            else:
                self.orders[oid] = (s, p, sz)
                self.levels[s][p][oid] = sz
        elif action == "R":
            self.clear()
        return True

    def queue_ahead(self, oid):
        """(qty ahead, orders ahead) of `oid` in its price queue, or (-1, -1)."""
        o = self.orders.get(oid)
        if o is None:
            return -1, -1
        qty = cnt = 0
        for k, v in self.levels[o[0]][o[1]].items():
            if k == oid:
                return qty, cnt
            qty += v
            cnt += 1
        return -1, -1

    def level(self, side, px):
        q = self.levels[side].get(px)
        return (sum(q.values()), len(q)) if q else (0, 0)

    def best(self, side):
        lv = self.levels[side]
        if not lv:
            return UNDEF_PRICE, 0, 0
        px, q = lv.peekitem(-1 if side == "B" else 0)
        return px, sum(q.values()), len(q)

    def to_frame(self):
        """Full L3 snapshot: one row per resting order, best levels first, FIFO within level."""
        rows = []
        for side, lv in self.levels.items():
            pxs = reversed(lv.keys()) if side == "B" else lv.keys()
            for lvl, px in enumerate(pxs):
                ahead = 0
                for pos, (oid, sz) in enumerate(lv[px].items()):
                    rows.append((side, lvl, px * PX_SCALE, pos, oid, sz, ahead))
                    ahead += sz
        return pd.DataFrame(rows, columns=["side", "level", "price", "queue_pos",
                                           "order_id", "size", "qty_ahead"])


L3_COLS = ["ts_recv", "ts_event", "sequence", "action", "side", "price", "size",
           "order_id", "flags", "is_last", "is_snapshot",
           "order_qty", "qty_ahead", "orders_ahead", "level_qty", "level_orders",
           "bid_px", "bid_qty", "bid_orders", "ask_px", "ask_qty", "ask_orders"]


def build_day(path, out):
    """Replay one day of MBO and write one row per event with book context.

    Book columns describe the book AFTER the event; they are only a consistent
    exchange state on rows with is_last=True (end of a CME event packet).
    Queue columns are for the event's own order: for C they are measured just
    before removal, for F at fill time (before the C/M that follows).
    """
    book = Book()
    cols = {k: [] for k in L3_COLS}
    unknown = 0
    for m in db.DBNStore.from_file(path):
        if not isinstance(m, db.MBOMsg):
            continue
        a, s, px, oid = str(m.action), str(m.side), m.price, m.order_id
        if a in ("C", "F"):
            qa, oa = book.queue_ahead(oid)
        if not book.apply(a, oid, s, px, m.size): # ingests book
            unknown += 1
        if a in ("A", "M"):
            qa, oa = book.queue_ahead(oid)
        elif a not in ("C", "F"):
            qa = oa = -1
        o = book.orders.get(oid) if a in ("A", "M", "C", "F") else None
        ls = {"B": "A", "A": "B"}.get(s) if a == "T" else s   # T side = aggressor
        lq, lo = book.level(ls, px) if ls in ("B", "A") and px != UNDEF_PRICE else (0, 0)
        bp, bq, bo = book.best("B")
        ap, aq, ao = book.best("A")
        f = int(m.flags)
        for k, v in zip(L3_COLS, (
                m.ts_recv, m.ts_event, m.sequence, a, s, px, m.size, oid, f,
                bool(f & db.RecordFlags.F_LAST), bool(f & db.RecordFlags.F_SNAPSHOT),
                o[2] if o else 0, qa, oa, lq, lo, bp, bq, bo, ap, aq, ao)):
            cols[k].append(v)

    def px_col(v):
        v = np.asarray(v, dtype=np.int64)
        return np.where(v == UNDEF_PRICE, np.nan, v * PX_SCALE).round(9)

    ts = lambda v: pa.array(np.asarray(v, dtype=np.int64), pa.timestamp("ns", tz="UTC"))
    i32 = lambda v: pa.array(v, pa.int32())
    table = pa.table({
        "ts_recv": ts(cols["ts_recv"]), "ts_event": ts(cols["ts_event"]),
        "sequence": pa.array(cols["sequence"], pa.uint32()),
        "action": pa.array(cols["action"]).dictionary_encode(),
        "side": pa.array(cols["side"]).dictionary_encode(),
        "price": px_col(cols["price"]), "size": pa.array(cols["size"], pa.uint32()),
        "order_id": pa.array(cols["order_id"], pa.uint64()),
        "flags": pa.array(cols["flags"], pa.uint8()),
        "is_last": cols["is_last"], "is_snapshot": cols["is_snapshot"],
        "order_qty": i32(cols["order_qty"]),
        "qty_ahead": i32(cols["qty_ahead"]), "orders_ahead": i32(cols["orders_ahead"]),
        "level_qty": i32(cols["level_qty"]), "level_orders": i32(cols["level_orders"]),
        "bid_px": px_col(cols["bid_px"]), "bid_qty": i32(cols["bid_qty"]),
        "bid_orders": i32(cols["bid_orders"]),
        "ask_px": px_col(cols["ask_px"]), "ask_qty": i32(cols["ask_qty"]),
        "ask_orders": i32(cols["ask_orders"]),
    })
    pq.write_table(table, out, compression="zstd")
    return table.num_rows, unknown


def cmd_build(sym, force=False):
    for path in sorted(RAW.glob(f"{sym}_mbo_*.dbn.zst")):
        out = L3 / path.name.replace(".dbn.zst", ".parquet")
        if out.exists() and not force:
            print("have", out.name)
            continue
        n, unknown = build_day(path, out)
        print(f"built {out.name}: {n:,} events, {unknown} refs to unknown orders")


def book_at(path, ts):
    """Order-level book at `ts` (UTC), finishing any packet still open at `ts`."""
    ts = pd.Timestamp(ts, tz="UTC").value
    book, done = Book(), True
    for m in db.DBNStore.from_file(path):
        if not isinstance(m, db.MBOMsg):
            continue
        if m.ts_recv > ts and done:
            break
        book.apply(str(m.action), m.order_id, str(m.side), m.price, m.size)
        done = bool(m.flags & db.RecordFlags.F_LAST)
    return book.to_frame()


def replay_mbo(path, my_price=None, my_side="B"):
    """Yield (ts_event, resting qty at my_price) at the end of each event packet."""
    book = Book()
    for m in db.DBNStore.from_file(path):
        if not isinstance(m, db.MBOMsg):
            continue
        book.apply(str(m.action), m.order_id, str(m.side), m.price, m.size)
        if m.flags & db.RecordFlags.F_LAST and my_price is not None:
            q = book.levels[my_side].get(my_price)
            if q:
                yield m.ts_event, sum(q.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["defs", "cost", "pull", "pull-days", "build", "snap"])
    p.add_argument("args", nargs="*")
    p.add_argument("--max-usd", type=float, default=0.0)
    p.add_argument("--force", action="store_true", help="build: overwrite existing parquet")
    a = p.parse_args()
    if a.cmd == "build":
        cmd_build(*a.args, force=a.force)
    elif a.cmd == "snap":
        sym, ts = a.args
        day = pd.Timestamp(ts).date()
        df = book_at(RAW / f"{sym}_mbo_{day}.dbn.zst", ts)
        pd.set_option("display.width", 200)
        print(df.groupby("side").head(8).to_string(index=False))
    else:
        c = client()
        if a.cmd == "defs":
            cmd_defs(c, a.args[0] if a.args else "2026-09-14")
        elif a.cmd == "cost":
            cmd_cost(c, *a.args)
        elif a.cmd == "pull":
            cmd_pull(c, *a.args, max_usd=a.max_usd)
        else:
            cmd_pull_days(c, *a.args, max_usd=a.max_usd)
