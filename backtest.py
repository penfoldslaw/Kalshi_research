"""Paper-trading backtest: Kalshi daily-high temperature brackets vs a weather-forecast model.
Walk-forward (each day's model uses only earlier days). No real orders, ever."""
import argparse, json, math, os, re, statistics, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import requests
import scanner as sc

CITIES = {  # series -> (lat, lon, tz)
    "KXHIGHNY": (40.7794, -73.9692, "America/New_York"),
}
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def cached(path, fn):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    d = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w", encoding="utf-8"))
    return d


def settled_events(series):
    out, cur = [], None
    while True:
        p = dict(series_ticker=series, status="settled", with_nested_markets="true", limit=200)
        if cur:
            p["cursor"] = cur
        d = sc.get("/events", **p)
        out += d["events"]
        cur = d.get("cursor")
        if not cur:
            return out


def event_date(tk):
    m = re.search(r"-(\d\d)([A-Z]{3})(\d\d)$", tk)
    return datetime(2000 + int(m[1]), MONTHS[m[2]], int(m[3])).date() if m else None


def bracket(m):
    """Inclusive integer-degree interval [lo, hi] a market pays YES on."""
    s = (m.get("yes_sub_title") or "").replace("°", "")
    if r := re.match(r"\s*(-?\d+)\s+or below", s): return (-999, int(r[1]))
    if r := re.match(r"\s*(-?\d+)\s+or above", s): return (int(r[1]), 999)
    if r := re.match(r"\s*(-?\d+)\s+to\s+(-?\d+)", s): return (int(r[1]), int(r[2]))
    return None


def forecasts(lat, lon, tz, start, end):
    """date -> daily max of the forecast issued ~1 day earlier (hourly previous_day1)."""
    d = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", timeout=60, params=dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end, timezone=tz,
        hourly="temperature_2m_previous_day1", temperature_unit="fahrenheit")).json()
    by = {}
    for t, v in zip(d["hourly"]["time"], d["hourly"]["temperature_2m_previous_day1"]):
        if v is not None:
            by.setdefault(t[:10], []).append(v)
    return {k: max(v) for k, v in by.items() if len(v) >= 20}


def candles(series, ticker, start_ts, end_ts):
    d = sc.get(f"/series/{series}/markets/{ticker}/candlesticks",
               start_ts=start_ts, end_ts=end_ts, period_interval=60)
    return d.get("candlesticks", [])


def quote_at(cs, snap_ts):
    """Last hourly candle ending at/before snap_ts -> (yes_ask, yes_bid)."""
    prior = [c for c in cs if c["end_period_ts"] <= snap_ts]
    if not prior:
        return None
    c = prior[-1]
    try:
        return float(c["yes_ask"]["close_dollars"]), float(c["yes_bid"]["close_dollars"])
    except (KeyError, TypeError, ValueError):
        return None


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob(lo, hi, mu, sd):
    a = -1e9 if lo <= -999 else (lo - 0.5 - mu) / sd
    b = 1e9 if hi >= 999 else (hi + 0.5 - mu) / sd
    return ncdf(b) - ncdf(a)


def fee10(p):
    """Per-contract taker fee when buying 10 at price p (ceil to cent per order)."""
    return math.ceil(round(0.07 * 10 * p * (1 - p) * 100, 6)) / 100 / 10


def build(series, snap_hour_utc, snap_days_before):
    lat, lon, tz = CITIES[series]
    evs = cached(f"cache/{series}_events.json", lambda: settled_events(series))
    rows = []
    for ev in evs:
        d = event_date(ev["event_ticker"])
        mk = [m for m in ev.get("markets", []) if m.get("result") in ("yes", "no") and bracket(m)]
        if d and len(mk) >= 4 and sum(m["result"] == "yes" for m in mk) == 1:
            rows.append((d, ev, mk))
    rows.sort(key=lambda r: r[0])
    fc = cached(f"cache/{series}_fc.json",
                lambda: forecasts(lat, lon, tz, str(rows[0][0]), str(rows[-1][0])))

    def work(item):
        d, ev, mk = item
        snap = datetime(d.year, d.month, d.day, snap_hour_utc, tzinfo=timezone.utc) - timedelta(days=snap_days_before)
        s_ts, e_ts = int(snap.timestamp()) - 6 * 3600, int(snap.timestamp()) + 3600
        out = []
        for m in mk:
            q = quote_at(candles(series, m["ticker"], s_ts, e_ts), int(snap.timestamp()))
            out.append(dict(ticker=m["ticker"], br=bracket(m), win=m["result"] == "yes", q=q))
        return dict(date=str(d), fc=fc.get(str(d)), mk=out)

    def run():
        with ThreadPoolExecutor(6) as ex:
            return list(ex.map(work, rows))
    return cached(f"cache/{series}_snap_{snap_days_before}d_{snap_hour_utc}z.json", run)


def simulate(days, margin, warm=20, window=60, shares=10):
    hist, trades, brier_m, brier_x, nb = [], [], 0.0, 0.0, 0
    for day in days:
        win_b = next((m["br"] for m in day["mk"] if m["win"]), None)
        usable = day["fc"] is not None and win_b and -999 < win_b[0] and win_b[1] < 999
        if day["fc"] is not None and len(hist) >= warm:
            errs = hist[-window:]
            bias, sd = statistics.mean(errs), max(1.5, statistics.pstdev(errs))
            mu = day["fc"] + bias
            for m in day["mk"]:
                if not m["q"]:
                    continue
                ask, bid = m["q"]
                if not (0 < ask <= 1 and 0 <= bid < 1 and bid < ask):
                    continue
                lo, hi = m["br"]
                p = prob(lo, hi, mu, sd)
                y = 1.0 if m["win"] else 0.0
                brier_m += (p - y) ** 2; brier_x += ((ask + bid) / 2 - y) ** 2; nb += 1
                if p - ask - fee10(ask) > margin and ask < 0.99:
                    trades.append(("YES", ask, y - ask - fee10(ask), p))
                elif bid - p - fee10(1 - bid) > margin and bid > 0.01:
                    trades.append(("NO", 1 - bid, (1 - y) - (1 - bid) - fee10(1 - bid), p))
        if usable:
            hist.append((win_b[0] + win_b[1]) / 2 - day["fc"])
    return trades, brier_m, brier_x, nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXHIGHNY")
    ap.add_argument("--snap-hour", type=int, default=22, help="UTC hour of the snapshot")
    ap.add_argument("--days-before", type=int, default=1)
    a = ap.parse_args()
    days = build(a.series, a.snap_hour, a.days_before)
    have = [d for d in days if d["fc"] is not None and any(m["q"] for m in d["mk"])]
    print(f"{len(days)} settled days, {len(have)} with forecast + quotes ({days[0]['date']} .. {days[-1]['date']})")
    print(f"\n{'margin':>7} {'trades':>7} {'win%':>6} {'avg$/ct':>8} {'total$@10':>10} {'ROI%':>7}")
    for margin in (0.0, 0.03, 0.05, 0.08, 0.12, 0.2):
        tr, bm, bx, nb = simulate(days, margin)
        if tr:
            pnl = sum(t[2] for t in tr); cost = sum(t[1] for t in tr)
            w = sum(t[2] > 0 for t in tr) / len(tr)
            print(f"{margin:>7.2f} {len(tr):>7} {w*100:>5.1f}% {pnl/len(tr):>8.3f} {pnl*10:>10.2f} {pnl/cost*100:>6.1f}%")
        else:
            print(f"{margin:>7.2f} {0:>7}")
    print(f"\nBrier (lower=better) over {nb} bracket-quotes:  model {bm/nb:.4f}   market mid {bx/nb:.4f}")
