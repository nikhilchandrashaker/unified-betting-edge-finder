# Unified Betting Edge Finder (NBA 2025-26, NFL 2025, Premier League 2025-26)

Run:  `python edge_finder.py run --data <folder with the 3 CSVs> --out .`
Add live games: `--upcoming new_games.csv` (same columns as the history files, results blank, new unique game ids).
Verify: `python edge_finder.py selftest --data <folder>` (settlement vs file labels, leakage test, null test).

## What the data is (and is not)
- Closing/opening lines and **no-vig consensus probabilities** plus final scores. No actual bookmaker prices, no player/injury data, one season per sport.
- So prices are ASSUMED: odds = 1 / (fair_prob x (1 + margin)); margin 4.5% spreads/totals/AH/OU/BTTS, 4% moneyline, 5% 1X2 (edit CFG["margin"]).
- "Bet at open" mode uses opening lines/probs and can measure CLV. "Bet at close" mode tests against the sharpest price; CLV is zero by construction there.

## Layers
1. Pattern analysis: segments (line buckets, team, rest, line movement, probability buckets). Walk-forward uses only prior games, shrinks residuals toward the market (k=40, min n=15). Full-sample scan with BH multiple-testing correction: results/pattern_scan_fullsample.csv.
2. ML: regularised logistic regression per market on market price + form/rest features, refit weekly on prior games only.
3. Strategy testing: edge = model prob - break-even prob; default threshold 2% (fixed before running; sensitivity table shown, not used for tuning). Staking: flat 1u (= 1% of a 100u bankroll) vs 25% Kelly capped at 3%.
   BOTH = both models must see positive edge; probability averaged.

## Files (results/)
summary_assumed_margin.csv, summary_zero_margin.csv (upper bound with no vig), summary_by_market.csv, baselines_naive.csv,
clv_by_market.csv, model_vs_market_logloss.csv, threshold_sensitivity.csv, pattern_scan_fullsample.csv, all_bets.csv, paper_picks.csv.
Equity chart: equity_curves.png.
