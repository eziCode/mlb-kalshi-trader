# Frozen probability, passive entry experiment — September 27, 2026

The frozen `corrected_01c` taker candidate trades about half the historical
games and has unproven small net gains. This separate prospective experiment
tests whether paying maker fees and joining displayed bids improves coverage
and net value. No maker profitability is inferred from the historical taker
fills. Parameters below are fixed before inspecting any live-game result.

Use the same frozen baseball and market-correction models, one-cent estimated
entry edge, five-second decision clock, innings 1–7, one-contract maximum
order, and maximum one filled opening order per game. Use both independent
team markets and a single $100 cash account. Pick the market with greatest
estimated edge at its bid; improve by one cent only when its spread is at
least three cents. New quotes must remain strictly below the observed ask.

Orders are post-only at modeled arrival. Submission and cancellation each
take 680ms, with separate 1.5/3-second stress replays. Queue ahead starts at
all displayed size at arrival. Only compatible public executions consume it;
book cancellations grant no automatic queue advancement. A trade through the
resting price can fill the exposed order, including during cancellation.
Trade exchange timestamps must also follow modeled arrival. Quote lifetime
is sixty seconds. Repricing waits for cancellation before a replacement is
submitted. No overlapping replacement quotes are assumed.

After the first partial fill, stop opening new orders and cancel the remainder
at the next decision; fills during cancellation still count. Hold acquired
contracts to settlement. Entry fee rate is 0.0175 with the current combined
principal/fee rounding. Reserve cash including rounding room. This may reduce
cost per fill but cannot make an adverse fill disappear.

Use the same received-state and pregame-anchor requirements as the frozen
forward taker policy. Missing states or an old feed cause cancellation of
opening quotes; pending cancellations remain exposed. Final reports retain
zero-trade games, partial quantities, actual market identities, fees, queue
waits, cancellation-race fills, and unsettled marked inventory. Credit payouts
only from the separate prospectively observed exchange-settlement journal.
These are shadow fills. Neither a touched price nor a positive mark establishes
that a real exchange order would have filled or earned money.
