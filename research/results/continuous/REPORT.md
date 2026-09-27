# Continuous strategy research

Frozen candidate: **revert_120_300s_00c**.

Development/selection passed: **False**. Live remains disabled.

One contract per entry; $100 initial cash per partition. Entries are simulated print-based fills, not verified fills.
These previously examined dates are chronological checks, not an untouched holdout.

## Frozen candidate

| Partition/scenario | Games | Entries | Entries/game | Games traded | Net PnL | Fees | Day-bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| development | 283 | 683 | 2.41 | 239 (84.5%) | $-37.84 | $16.42 | [$-42.80, $-32.79] |
| selection | 186 | 435 | 2.34 | 152 (81.7%) | $-17.81 | $9.92 | [$-24.13, $-11.37] |
| final/reference | 141 | 367 | 2.60 | 123 (87.2%) | $-12.72 | $8.76 | [$-18.40, $-6.75] |
| final/latency_1_5s | 141 | 353 | 2.50 | 122 (86.5%) | $-18.31 | $8.52 | [$-24.53, $-12.50] |
| final/latency_3s | 141 | 354 | 2.51 | 125 (88.7%) | $-16.15 | $8.48 | [$-23.01, $-9.46] |
| final/two_cent_cost | 141 | 286 | 2.03 | 117 (83.0%) | $-15.53 | $6.86 | [$-21.71, $-9.61] |
| final/one_percent_volume | 141 | 200 | 1.42 | 95 (67.4%) | $-9.77 | $4.48 | [$-16.59, $-3.16] |
| final/fifty_ms_window | 141 | 137 | 0.97 | 86 (61.0%) | $-7.52 | $3.11 | [$-11.10, $-3.80] |
| final/publication_plus_2s | 141 | 366 | 2.60 | 123 (87.2%) | $-12.53 | $8.74 | [$-18.27, $-6.49] |

## All trials

Every declared candidate and both forced-activity controls appear below.

| Candidate | Dev PnL | Selection PnL | Selection entries/game | Selection game coverage |
| --- | ---: | ---: | ---: | ---: |
| revert_30_60s_00c | $-21.66 | $-12.87 | 1.42 | 65.1% |
| revert_30_60s_01c | $-18.19 | $-11.07 | 1.16 | 59.1% |
| revert_30_60s_03c | $-11.29 | $-6.50 | 0.78 | 47.3% |
| revert_30_300s_00c | $-16.64 | $-9.30 | 1.21 | 65.1% |
| revert_30_300s_01c | $-15.86 | $-9.44 | 1.02 | 59.1% |
| revert_30_300s_03c | $-10.53 | $-6.68 | 0.70 | 47.3% |
| revert_120_60s_00c | $-49.26 | $-26.74 | 2.93 | 81.7% |
| revert_120_60s_01c | $-43.17 | $-21.55 | 2.49 | 78.5% |
| revert_120_60s_03c | $-32.92 | $-17.40 | 1.84 | 71.5% |
| revert_120_300s_00c | $-37.84 | $-17.81 | 2.34 | 81.7% |
| revert_120_300s_01c | $-34.66 | $-15.92 | 2.01 | 78.5% |
| revert_120_300s_03c | $-26.50 | $-12.38 | 1.52 | 71.5% |
| momentum_30_60s_00c | $-9.55 | $-7.05 | 0.68 | 37.1% |
| momentum_30_60s_01c | $-5.03 | $-4.34 | 0.52 | 30.6% |
| momentum_30_60s_03c | $-3.42 | $-3.79 | 0.31 | 21.5% |
| momentum_30_300s_00c | $-9.74 | $-5.57 | 0.61 | 37.1% |
| momentum_30_300s_01c | $-4.79 | $-3.03 | 0.48 | 30.6% |
| momentum_30_300s_03c | $-2.89 | $-3.47 | 0.30 | 21.5% |
| momentum_120_60s_00c | $-39.93 | $-22.64 | 2.34 | 73.7% |
| momentum_120_60s_01c | $-31.85 | $-19.36 | 1.97 | 67.7% |
| momentum_120_60s_03c | $-23.94 | $-12.48 | 1.39 | 56.5% |
| momentum_120_300s_00c | $-32.04 | $-18.01 | 1.84 | 73.7% |
| momentum_120_300s_01c | $-25.57 | $-16.50 | 1.56 | 67.7% |
| momentum_120_300s_03c | $-19.75 | $-11.34 | 1.14 | 56.5% |
| cross_market_60s_00c | $-0.23 | $0.38 | 0.01 | 1.1% |
| cross_market_60s_01c | $0.00 | $0.00 | 0.00 | 0.0% |
| cross_market_60s_03c | $0.00 | $0.00 | 0.00 | 0.0% |
| cross_market_300s_00c | $-0.27 | $0.38 | 0.01 | 1.1% |
| cross_market_300s_01c | $0.00 | $0.00 | 0.00 | 0.0% |
| cross_market_300s_03c | $0.00 | $0.00 | 0.00 | 0.0% |
| state_value_60s_00c | $-92.44 | $-50.73 | 5.68 | 94.6% |
| state_value_60s_01c | $-84.17 | $-47.37 | 5.17 | 92.5% |
| state_value_60s_03c | $-72.30 | $-39.54 | 4.40 | 86.6% |
| state_value_300s_00c | $-65.95 | $-35.26 | 4.13 | 94.6% |
| state_value_300s_01c | $-59.53 | $-33.39 | 3.78 | 92.5% |
| state_value_300s_03c | $-52.06 | $-26.64 | 3.23 | 86.6% |
| all_play_60s_00c | $-4.56 | $-2.73 | 0.29 | 19.9% |
| all_play_60s_01c | $-4.43 | $-1.57 | 0.23 | 16.7% |
| all_play_60s_03c | $-1.16 | $-1.48 | 0.16 | 12.4% |
| all_play_300s_00c | $-6.87 | $-2.05 | 0.28 | 19.9% |
| all_play_300s_01c | $-5.64 | $-0.85 | 0.22 | 16.7% |
| all_play_300s_03c | $-1.37 | $-1.19 | 0.16 | 12.4% |
| learned_60s_00c | $0.00 | $0.00 | 0.00 | 0.0% |
| learned_60s_01c | $0.00 | $0.00 | 0.00 | 0.0% |
| learned_60s_03c | $0.00 | $0.00 | 0.00 | 0.0% |
| learned_300s_00c | $1.20 | $-0.95 | 0.05 | 5.4% |
| learned_300s_01c | $1.26 | $-0.71 | 0.05 | 4.8% |
| learned_300s_03c | $0.77 | $0.04 | 0.02 | 1.6% |
| activity_control_60s_-1000c | $-95.95 | $-74.45 | 8.35 | 100.0% |
| activity_control_300s_-1000c | $-85.41 | $-50.17 | 5.91 | 100.0% |

## Interpretation limits

The price proxies are recent same-side trades, not historical bid/ask quotes. Print-based fills cannot establish queue, depth, or availability at order arrival. Maker fills are never inferred.

Final MLB archives can contain corrections. State reception is modeled; the terminal latency profile came from one July sample. Final game boundaries delimit the available archive and unfilled exits settle.

Bootstrap intervals describe day variation under these assumptions. They do not correct multiple testing, earlier inspection of this period, or execution-model error. The final period contains only 11 days.

Cash is reserved for submissions, all games share one chronological portfolio, and settlement proceeds are withheld. Each partition starts with a fresh $100; partition returns must not be added as a compounded live path.

The frequency goal is reported as both entries/game and game coverage. No failed candidate is enabled or described as profitable live.
