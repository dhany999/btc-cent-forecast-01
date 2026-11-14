# btc-cent-forecast-01 — winsorized mean-reversion forecaster

Entry for make.fun competition `btc-cent-forecast-01`: forecast Bitcoin's
price 240 minutes after each checkpoint. Python 3.13, standard library only,
no network, deterministic.

## Approach — winsorized mean reversion toward the 24h centre

Work in log price. The anchor is the **winsorized mean** of the log prices
over the last 1,440 minutes (24 h): each log price is clipped to the window
mean ± 3 standard deviations before averaging, so one pool-jump minute (a
large trade moving this single Orca pool's price for a moment) cannot drag
the anchor. The forecast pulls the checkpoint price 30% of the way back
toward that centre:

```
forecast = last_price * exp(0.30 * (anchor - log(last_price)))
```

clamped to a ±5% log move. There is **no momentum term**: on 528 in-series
checkpoints, momentum of every window (60–1,440 min) and shrinkage made the
error worse, while this reversion beat persistence overall on the larger
in-series backtests (−3.66% on 66 checkpoints, −4.77% on 528), though it
trails persistence on the 5 gradeable official practice cases (see table
below). The forecast is the **median** of the predictive
distribution — MAE is a proper scoring rule for one number, so the median is
the best guess. Fewer than 60 prices falls back to pure persistence. Prices
are rounded to the grid's 5 decimals and clamped inside (0, 10,000,000).

The pattern is adapted from the `polymarket-nvidia-ceo-forecast-01` entry's
"Winsorized sigma forecaster" (winsorized recent changes, shrunk drift
toward a centre, median forecast) — rebuilt from scratch for a single price
series and a median/MAE target; no code was copied.

## Local practice grade

Each practice case's actual target = the next case's last close, so p00–p04
are gradeable locally (p05's actual is not in the kit). The 5-case sample is
tiny and dominated by one downtrend leg, so the parameter was chosen on two
larger in-series backtests (checkpoints inside each case's 72 h window,
target = checkpoint + 240 min) and the official 5-case numbers are reported
alongside, honestly, for comparison.

| evaluation | this entry | persistence baseline |
|---|---|---|
| official 5 cases (p00–p04) | **49,556.06 c** (mean $495.56, worst $851.48) | 47,054.66 c (mean $470.55, worst $766.97) |
| 66 non-overlapping 240-min checkpoints | **39,452.55 c** | 40,950.12 c (−3.66%) |
| 528 dense 30-min checkpoints | **41,732.18 c** | 43,821.74 c (−4.77%) |

Per-case errors on the official 5: p00 $286.0, p01 $851.5, p02 $770.7,
p03 $196.0, p04 $373.7. The reversion pull loses on p01/p02 (mid-downtrend
checkpoints, where the price kept falling) and wins big on p03 ($196 vs
$567 for persistence). The parameter sits on a flat optimum: REV_FRAC
0.25–0.45 all beat persistence by ~3–5% on the larger backtests, though the
entry trails persistence on the 5 gradeable official cases (see table above).

Caveat, as the rules say: a practice score is no guide to the final — the
final is scored on unseen prices, and a prize needs a score strictly below
the baseline's on the final.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~13 ms per case locally (limit is 8 s), output < 1 KiB per case.
Two runs produce byte-identical output (verified).
