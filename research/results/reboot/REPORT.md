# Frozen-policy later-period diagnostic

2026-08-12 through 2026-09-26; $100.00 starting cash.

Historical executions are a fill proxy. This report does not authorize deployment.

| Scenario | Fills | Net proxy PnL | Day-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| bounded_reference | 24 | $6.73 | $1.20 to $13.86 |
| latency_1_5s | 17 | $-0.66 | $-3.98 to $2.53 |
| latency_3s | 26 | $-6.10 | $-14.21 to $1.58 |
| two_cent_cost | 11 | $1.93 | $-1.06 to $5.58 |
| one_percent_volume | 24 | $2.79 | $0.14 to $6.57 |
| fifty_ms_window | 7 | $2.41 | $-0.13 to $6.65 |
| publication_plus_2s | 25 | $-2.45 | $-7.70 to $1.44 |

- No historical quotes, depth, queue position, or actual IOC fills
- Final MLB archives may include later official corrections; reception timestamps are modeled
- Publication-delay profile is from one July sample; fixed costs/windows are stress assumptions
- Earlier development and live monitoring mean later dates are not automatically a blind holdout
- Settlement cash is withheld throughout the replay; partial exits release cash only at final exit
- Cash admission filters proposed fills; pending-order reservations and replacement signals after cash rejection are not simulated
- Positive proxy PnL never enables deployment
