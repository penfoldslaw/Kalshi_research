"""Read-only Kalshi scanner: multi-outcome events where buying YES on every
outcome costs less than $1 after fees. Never places orders."""
import argparse, csv, math, os, time
from datetime import datetime, timezone
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
BASE_FEE = 0.07
OPP_FIELDS = ["ts", "event_ticker", "title", "legs", "n", "cost", "fees", "payout",
              "net", "roi_pct", "days_to_settle", "roi_pct_per_day", "top_sum",
              "exhaustive_verified", "fee_type", "recheck_nets", "stable", "outcomes", "rules"]
SCAN_FIELDS = ["ts", "events_total", "events_mutex", "events_top_sum_lt_1",
               "events_net_positive", "min_top_sum", "seconds"]

s = requests.Session()
_series_fee = {}


def get(path, **params):
    for attempt in range(5):
        r = s.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited: {path}")


def fetch_events():
    cursor = None
    while True:
        p = dict(status="open", with_nested_markets="true", limit=200)
        if cursor:
            p["cursor"] = cursor
        d = get("/events", **p)
        yield from d["events"]
        cursor = d.get("cursor")
        if not cursor:
            return


def fee_info(series_ticker):
    if series_ticker not in _series_fee:
        try:
            ser = get(f"/series/{series_ticker}")["series"]
            _series_fee[series_ticker] = (ser.get("fee_type"), float(ser.get("fee_multiplier") or 1))
        except Exception:
            _series_fee[series_ticker] = (None, 1.0)
    return _series_fee[series_ticker]


def order_fee(fills, mult):
    """Taker fee for one order: ceil_to_cent(0.07*mult*sum(n*P*(1-P)))."""
    raw = BASE_FEE * mult * sum(q * p * (1 - p) for p, q in fills)
    return math.ceil(round(raw * 100, 6)) / 100


def ask_ladder(ticker):
    """YES asks derived from NO bids: ask = 1 - no_bid, same size. Cheapest first."""
    ob = get(f"/markets/{ticker}/orderbook")["orderbook_fp"]
    lv = [(round(1 - float(p), 4), float(q)) for p, q in ob.get("no_dollars") or []]
    return sorted(lv)


def walk(ladder, n):
    """Fills (price, qty) to buy n contracts, or None if not enough depth."""
    fills, need = [], n
    for p, q in ladder:
        take = min(q, need)
        fills.append((p, take))
        need -= take
        if need <= 0:
            return fills
    return None


def best_size(ladders, mult, bankroll):
    best = None
    max_n = int(min(sum(q for _, q in l) for l in ladders))
    for n in range(1, max_n + 1):
        legs = [walk(l, n) for l in ladders]
        cost = sum(p * q for f in legs for p, q in f)
        if cost > bankroll:
            break
        fees = sum(order_fee(f, mult) for f in legs)
        net = n - cost - fees
        roi = net / (cost + fees)
        if best is None or net > best["net"]:
            best = dict(n=n, cost=cost, fees=fees, net=net, roi=roi)
    return best


def parse_ts(x):
    try:
        return datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None


def days_to_settle(markets):
    ts = [parse_ts(m.get("expected_expiration_time") or "") for m in markets]
    ts = [t for t in ts if t]
    if not ts:
        return None
    return max(0.01, (max(ts) - datetime.now(timezone.utc)).total_seconds() / 86400)


def append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        w.writerow(row)


def recheck(mk, mult, bankroll, times, gap):
    """Re-pull fresh order books a few times; return the net of each pass."""
    nets = []
    for i in range(times):
        if i:
            time.sleep(gap)
        b = best_size([ask_ladder(m["ticker"]) for m in mk], mult, bankroll)
        nets.append(round(b["net"], 2) if b else 0.0)
    return nets


def scan(bankroll, near, min_sum, times, gap, out_dir):
    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = mutex = lt1 = pos = 0
    min_seen = None
    rows = []
    for ev in fetch_events():
        total += 1
        if not ev.get("mutually_exclusive"):
            continue
        allm = ev.get("markets", [])
        mk = [m for m in allm if m.get("status") == "active"]
        if len(mk) < 2:
            continue
        if any(m.get("status") not in ("active", "finalized", "settled") for m in allm):
            continue  # an outcome we can't buy (e.g. inactive) breaks the lock
        try:
            top = [float(m["yes_ask_dollars"]) for m in mk]
        except (KeyError, ValueError, TypeError):
            continue
        if any(a <= 0 or a >= 1 for a in top):
            continue  # a leg with no ask is not executable
        mutex += 1
        top_sum = sum(top)
        min_seen = top_sum if min_seen is None else min(min_seen, top_sum)
        if top_sum >= near or top_sum < min_sum:
            continue
        lt1 += top_sum < 1
        fee_type, mult = fee_info(ev["series_ticker"])
        if fee_type not in ("quadratic", "quadratic_with_maker_fees"):
            continue  # unknown fee model: don't guess
        ladders = [ask_ladder(m["ticker"]) for m in mk]
        best = best_size(ladders, mult, bankroll)
        if not best or best["net"] <= 0:
            continue
        pos += 1
        d = days_to_settle(mk)
        nets = recheck(mk, mult, bankroll, times, gap)
        stable = all(n > 0 for n in nets)
        rules = (mk[0].get("rules_primary") or "").replace(chr(10), " ")[:400]
        outcomes = " | ".join(m.get("yes_sub_title") or m["ticker"] for m in mk)[:300]
        rows.append(dict(ts=ts, event_ticker=ev["event_ticker"], title=ev.get("title"),
                   legs=len(mk), n=best["n"], cost=round(best["cost"], 2),
                   fees=round(best["fees"], 2), payout=best["n"],
                   net=round(best["net"], 2), roi_pct=round(best["roi"] * 100, 2),
                   days_to_settle=round(d, 2) if d else "",
                   roi_pct_per_day=round(best["roi"] * 100 / d, 3) if d else "",
                   top_sum=round(top_sum, 4),
                   exhaustive_verified="NO - check rules", fee_type=fee_type,
                   recheck_nets="/".join(map(str, nets)), stable="YES" if stable else "NO",
                   outcomes=outcomes, rules=rules))
    rows.sort(key=lambda r: -r["top_sum"])
    for r in rows:
        append(os.path.join(out_dir, "opportunities.csv"), OPP_FIELDS, r)
        print(f"  sum={r['top_sum']:<6} legs={r['legs']:<3} n={r['n']:<4} net=${r['net']:<6} "
              f"roi={r['roi_pct']}% days={r['days_to_settle']} stable={r['stable']} "
              f"nets={r['recheck_nets']}  {r['event_ticker']}  {r['title']}")
    row = dict(ts=ts, events_total=total, events_mutex=mutex, events_top_sum_lt_1=lt1,
               events_net_positive=pos, min_top_sum=round(min_seen, 4) if min_seen else "",
               seconds=round(time.time() - t0, 1))
    append(os.path.join(out_dir, "scans.csv"), SCAN_FIELDS, row)
    print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=30.0)
    ap.add_argument("--near", type=float, default=1.02,
                    help="fetch order books for events whose top-of-book ask sum is below this")
    ap.add_argument("--min-sum", type=float, default=0.90,
                    help="ignore events whose ask sum is below this (incomplete outcome lists)")
    ap.add_argument("--rechecks", type=int, default=3, help="fresh order-book re-pulls per candidate")
    ap.add_argument("--recheck-gap", type=float, default=3.0, help="seconds between re-pulls")
    ap.add_argument("--loop", type=int, default=0, help="seconds between scans (0 = once)")
    ap.add_argument("--out", default="logs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    while True:
        scan(a.bankroll, a.near, a.min_sum, a.rechecks, a.recheck_gap, a.out)
        if not a.loop:
            break
        time.sleep(a.loop)
