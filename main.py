#!/usr/bin/env python3
"""btc-cent-forecast-01: forecast BTC's price 240 minutes ahead (Python 3.13,
stdlib only, no network, deterministic).

Method — winsorized mean-reversion forecaster (adapted from the
"Winsorized sigma" pattern of the polymarket-nvidia-ceo-forecast-01 entry,
rebuilt for a single price series and a median/MAE target):

  * Work in log price. The anchor is the WINSORIZED MEAN of the log prices
    over the last 1440 minutes (24 h): each log price is clipped to the
    window mean +/- 3 standard deviations before averaging, so one
    pool-jump minute (a large trade moving this single Orca pool's price
    for a moment) cannot drag the anchor.
  * Forecast = last_price * exp(REV_FRAC * (anchor - log(last_price))):
    the checkpoint is pulled REV_FRAC = 30% of the way back toward its
    24-hour centre. BTC's 4h moves are dominated by noise, so there is no
    momentum term at all: on 528 in-series checkpoints momentum of every
    window and shrinkage made the error WORSE, while this reversion beat
    persistence overall on the larger in-series backtests (-3.66% on 66
    checkpoints, -4.77% on 528), though it trails persistence on the 5
    gradeable official practice cases (see README).
  * The forecast is the MEDIAN of the predictive distribution (MAE is a
    proper scoring rule for one number, so the median is the best guess),
    with the log move clamped to +/-5% so no case can run away.
  * Fallback: fewer than 60 prices -> pure persistence (the house
    baseline). Prices are rounded to the grid's 5 decimals and clamped
    inside (0, 10,000,000).

Reads every *.in file in the input directory (name order) and writes one
*.out file per case to the output directory. Input/output dirs come from
$MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR, else /input / /output.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# --- tunables (chosen on 66 non-overlapping + 528 dense in-series
# checkpoints; the curve is flat for REV_FRAC 0.25-0.45, so 0.30 is a
# robust middle, not a sharp optimum) ---
REV_WINDOW = 1440      # minutes whose log prices form the anchor
REV_FRAC = 0.30        # fraction of the (anchor - last) gap pulled back
WINSOR_SIGMA = 3.0     # clip log prices to mean +/- this many sd
CLAMP_LOG = math.log(1.05)  # max |log move| of the forecast vs last price
MIN_PRICES = 60


def _price5(x):
    """Clamp into (0, 10,000,000) and round to the grid's 5 decimals."""
    v = round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5
    return v + 0.0


def _winsorized_mean_log(logs, window, k):
    """Mean of the last `window` log prices, each clipped to the window
    mean +/- k standard deviations (pool jumps cannot drag the anchor)."""
    w = logs[-window:]
    n = len(w)
    mean = sum(w) / n
    var = sum((x - mean) ** 2 for x in w) / n
    sd = math.sqrt(var)
    if sd <= 0:
        return mean
    lo, hi = mean - k * sd, mean + k * sd
    total = 0.0
    for x in w:
        if x < lo:
            total += lo
        elif x > hi:
            total += hi
        else:
            total += x
    return total / n


def forecast_price(prices, horizon=240, params=None):
    """Median forecast of the price `horizon` minutes after the last close.

    `horizon` is accepted for interface symmetry with the kit (240 always);
    the reversion pull is a fraction of the gap, not a per-minute drift,
    so it does not scale with horizon.
    """
    p = dict(rev_window=REV_WINDOW, rev_frac=REV_FRAC,
             winsor_sigma=WINSOR_SIGMA, clamp_log=CLAMP_LOG)
    if params:
        p.update(params)
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not (isinstance(last, (int, float)) and math.isfinite(last)) or last <= 0:
        raise ValueError("last price is not a finite positive number")
    if len(prices) < MIN_PRICES:
        return _price5(last)

    logs = [math.log(x) for x in prices]
    anchor = _winsorized_mean_log(logs, p["rev_window"], p["winsor_sigma"])
    log_move = p["rev_frac"] * (anchor - logs[-1])
    if log_move > p["clamp_log"]:
        log_move = p["clamp_log"]
    elif log_move < -p["clamp_log"]:
        log_move = -p["clamp_log"]
    return _price5(last * math.exp(log_move))


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError("%s does not hold exactly one market" % path.name)
        price = forecast_price(markets[0]["prices"], int(case.get("horizon_minutes", 240)))
        out_path = out_dir / ("%s.out" % case_id)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": [{"case": case_id, "price": price}]}) + "\n")


if __name__ == "__main__":
    main()
