"""Read-only Kalshi scanner for nested-threshold locks.

For markets on one quantity at different cutoffs ("above 75" vs "above 80"),
the higher cutoff can only resolve YES if the lower one does. So buying
YES(lower) + NO(higher) pays at least $1 in every outcome ($2 if the value
lands in between). If that pair costs < $1 after fees, it's a lock.
Never places orders."""
import argparse, itertools, json, os, time
from datetime import datetime, timezone
import scanner as sc

OPP_FIELDS = ["ts", "event_ticker", "title", "kind", "yes_leg", "no_leg", "n", "cost", "fees",
              "net", "roi_pct", "days_to_settle", "roi_pct_per_day", "top_cost",
              "recheck_nets", "stable", "fee_type", "rules"]
UP = ("greater", "greater_or_equal")
DOWN = ("less", "less_or_equal")


def strike(m):
    return m.get("floor_strike") if m.get("strike_type") in UP else m.get("cap_strike")


def families(ev):
    """Yield lists of nested markets, index 0 = SUPERSET (the one that is YES in
    more outcomes). Two kinds, both requiring strictly different thresholds:
      strike:   one-sided strikes only (floor XOR cap), same custom_strike & expiry.
                greater*: lower floor is the superset; less*: higher cap is.
      deadline: same strike, labels start with 'By'/'Before', different close_time.
                the later deadline is the superset.
    Anything ambiguous (e.g. exact-count brackets labelled 'less') is skipped."""
    strike_g, dead_g = {}, {}
    for m in ev.get("markets", []):
        st = m.get("strike_type")
        if st not in UP + DOWN or m.get("status") != "active":
            continue
        cs = json.dumps(m.get("custom_strike"), sort_keys=True)
        one_sided = (m.get("cap_strike") is None) if st in UP else (m.get("floor_strike") is None)
        if one_sided and strike(m) is not None:
            strike_g.setdefault((st, cs, m.get("expected_expiration_time")), []).append(m)
        label = (m.get("yes_sub_title") or "").lower()
        if label.startswith(("by ", "before ")) and m.get("close_time"):
            dead_g.setdefault((st, str(strike(m)), cs), []).append(m)
    for key, ms in strike_g.items():
        ms = [m for m in ms]
        if len({strike(m) for m in ms}) >= 2:
            yield "strike", sorted(ms, key=strike, reverse=key[0] in DOWN)
    for key, ms in dead_g.items():
        if len({m["close_time"] for m in ms}) >= 2:
            yield "deadline", sorted(ms, key=lambda m: m["close_time"], reverse=True)


def distinct(kind, a, b):
    if kind == "strike":
        return strike(a) != strike(b)
    return a["close_time"] != b["close_time"]


def f(m, k):
    try:
        return float(m[k])
    except (KeyError, TypeError, ValueError):
        return None


_book_cache = {}


def books(m, fresh=False):
    """(yes_ask_ladder, no_ask_ladder), each cheapest-first, from the order book.
    First-pass calls reuse a per-scan cache; rechecks pass fresh=True."""
    if not fresh and m["ticker"] in _book_cache:
        return _book_cache[m["ticker"]]
    ob = sc.get(f"/markets/{m['ticker']}/orderbook")["orderbook_fp"]
    yes_asks = sorted((round(1 - float(p), 4), float(q)) for p, q in ob.get("no_dollars") or [])
    no_asks = sorted((round(1 - float(p), 4), float(q)) for p, q in ob.get("yes_dollars") or [])
    _book_cache[m["ticker"]] = (yes_asks, no_asks)
    return yes_asks, no_asks


def evaluate(sup, sub, mult, bankroll, fresh=False):
    """Buy YES(superset) + NO(subset)."""
    yes_l = books(sup, fresh)[0]
    no_h = books(sub, fresh)[1]
    if not yes_l or not no_h:
        return None
    return sc.best_size([yes_l, no_h], mult, bankroll)


def scan(bankroll, slack, times, gap, out_dir, cache, recheck_top):
    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if cache and os.path.exists(cache):
        events = json.load(open(cache))
    else:
        events = list(sc.fetch_events())
    pairs = fams = 0
    rows = []
    for ev in events:
        for kind, fam in families(ev):
            fams += 1
            for sup, sub in itertools.combinations(fam, 2):
                if not distinct(kind, sup, sub):
                    continue
                pairs += 1
                ask_l, bid_h = f(sup, "yes_ask_dollars"), f(sub, "yes_bid_dollars")
                if ask_l is None or bid_h is None or not (0 < ask_l < 1) or bid_h <= 0:
                    continue
                top_cost = ask_l + (1 - bid_h)
                if top_cost >= 1 + slack:
                    continue
                fee_type, mult = sc.fee_info(ev["series_ticker"])
                if fee_type not in ("quadratic", "quadratic_with_maker_fees"):
                    continue
                best = evaluate(sup, sub, mult, bankroll)
                if not best or best["net"] <= 0:
                    continue
                d = sc.days_to_settle([sup, sub])
                rows.append(dict(
                    ts=ts, event_ticker=ev["event_ticker"], title=ev.get("title"),
                    kind=kind, yes_leg=f"YES {sup['ticker']} ({sup.get('yes_sub_title')})",
                    no_leg=f"NO {sub['ticker']} ({sub.get('yes_sub_title')})",
                    n=best["n"], cost=round(best["cost"], 2), fees=round(best["fees"], 2),
                    net=round(best["net"], 2), roi_pct=round(best["roi"] * 100, 2),
                    days_to_settle=round(d, 2) if d else "",
                    roi_pct_per_day=round(best["roi"] * 100 / d, 3) if d else "",
                    top_cost=round(top_cost, 4), recheck_nets="", stable="not rechecked",
                    fee_type=fee_type,
                    rules=(sup.get("rules_primary") or "").replace(chr(10), " ")[:200],
                    _m=(sup, sub, mult)))
    rows.sort(key=lambda r: -r["net"])
    print(f"first pass: {len(rows)} pairs positive after fees; rechecking top {recheck_top}", flush=True)
    for r in rows[:recheck_top]:
        sup, sub, mult = r["_m"]
        nets = []
        for i in range(times):
            time.sleep(gap if i else 0)
            b = evaluate(sup, sub, mult, bankroll, fresh=True)
            nets.append(round(b["net"], 2) if b else 0.0)
        r["recheck_nets"] = "/".join(map(str, nets))
        r["stable"] = "YES" if all(n > 0 for n in nets) else "NO"
    for r in rows:
        r.pop("_m")
    for r in rows:
        sc.append(os.path.join(out_dir, "threshold_opportunities.csv"), OPP_FIELDS, r)
        print(f"  net=${r['net']:<6} n={r['n']:<4} roi={r['roi_pct']}% days={r['days_to_settle']} "
              f"stable={r['stable']} nets={r['recheck_nets']}  {r['yes_leg']}  +  {r['no_leg']}")
    print(dict(ts=ts, events=len(events), families=fams, pairs_checked=pairs,
               locks_found=len(rows), seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=30.0)
    ap.add_argument("--slack", type=float, default=0.0,
                    help="fetch books when top-of-book cost < 1+slack (cached quotes may be stale)")
    ap.add_argument("--recheck-top", type=int, default=15, help="recheck only the N best pairs")
    ap.add_argument("--rechecks", type=int, default=3)
    ap.add_argument("--recheck-gap", type=float, default=2.0)
    ap.add_argument("--cache", default="", help="events JSON to reuse instead of refetching")
    ap.add_argument("--out", default="logs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scan(a.bankroll, a.slack, a.rechecks, a.recheck_gap, a.out, a.cache, a.recheck_top)
