# Trap Trading

## ADANIGREEN Case Study

This note captures the earlier analysis of `ADANIGREEN` on the `30-minute` timeframe for `2026-04-02`, starting from the first candle of the session.

The core read was:

- this was not a random breakout
- it behaved like a **failed bearish expansion**
- that failure later turned into a **short trap and squeeze**

The most important lesson from the day is that the best entry was not the late obvious breakout. The better entry came when a heavy bearish candle failed to produce downside follow-through and its liquidity level got reclaimed.

## 30-Min Timeline

### Opening phase

- First `30-minute` candle: `840 -> 823.8`
- Candle low: `814.75`
- This was a wide opening flush and looked bearish on face value.

Even after that opening weakness, price did not meaningfully continue below `814.75` during the session. That mattered because it suggested that the early selloff was being absorbed rather than accepted lower.

### Midday trap formation

The key anomaly came on the `11:45` candle:

- Open: `827.05`
- Close: `822.5`
- High: `827.5`
- Low: `820.1`
- Volume: `5,678,389`

Important context:

- this volume was about `25.9x` the average of the prior `5` bars
- the candle was red and visually looked like bearish continuation
- but the next candles failed to extend the downside

That is the trap signature:

- heavy bearish participation appears
- price looks weak
- but the market cannot continue lower

### Reclaim phase

After the high-volume bearish bar:

- `13:15` closed near `826.0`
- `13:45` closed at `837.0`

The `11:45` bearish trap bar high was `827.5`.

Once price reclaimed that level, the bearish thesis failed. That reclaim was the actual pivot point. The market was no longer just bouncing inside weakness; it was actively invalidating the strongest bearish candle of the day.

### Expansion phase

Later in the session:

- `14:45` closed at `859.85`
- Day high reached `863.95`

Actionable move:

- `13:15 close 826.0` to `14:45 close 859.85`
- Move size: `+33.85` points in about `90` minutes

Full intraday swing:

- Day low `814.75`
- Day high `863.95`
- Swing size `+49.2` points

## Why The Move Happened

### 1. Opening selloff was absorbed

The first candle was aggressive, but the session never truly accepted prices below the opening low. That usually means early supply was met by stronger demand.

### 2. The `11:45` candle was capitulation, not continuation

The `11:45` bar looked bearish, but it did not produce real downside follow-through. When a huge red candle fails to continue lower, it often marks exhaustion rather than trend strength.

### 3. Futures were stronger than spot

Even while spot looked weak, derivatives were not confirming deep weakness:

- around `11:45`, futures vs spot stayed near `+0.55%`
- around `13:15`, futures vs spot stayed near `+0.58%`

This divergence mattered. Spot looked bearish, but futures premium suggested the broader positioning was not aligned with a clean downside continuation.

### 4. Options turned bullish before the final breakout

During the reversal phase:

- `13:15`: call expansion about `+8.46%`, put decay about `-7.54%`
- `13:45`: call expansion about `+15.09%`, put decay about `-9.10%`
- `14:45`: call expansion about `+21.05%`, put decay about `-21.36%`

This is the exact options behavior expected before a strong upside move:

- calls gaining strength
- puts losing value
- derivatives rotating before spot completes the visual breakout

### 5. Bearish liquidity got reclaimed

The most important level was not the broad session high first. It was the high of the bearish trap candle: `827.5`.

Once that level was reclaimed, the strongest bearish bar of the day had failed. That was the structural clue.

### 6. The obvious breakout came later

Broader session resistance was around `841.8`.

That level breaking was useful confirmation, but not the earliest edge. Waiting only for that breakout would have caught the move late. The earlier edge came from the trap failure itself.

## Best Early Entry

The best early long was the reclaim of the bearish trap bar, not the later clean breakout above broader session resistance.

Practical sequence:

1. Identify a bearish bias candle.
2. Confirm that it is high-volume and abnormal.
3. Watch whether price fails to continue lower over the next few candles.
4. Track whether futures remain firm and options begin to flip bullish.
5. Trigger long when price closes back above the bearish trap candle high.

For `ADANIGREEN` on `2026-04-02`:

- trap bar high: `827.5`
- reclaim developed through the `13:15` to `13:45` sequence
- that reclaim was the early signal
- the later `841.8` breakout was confirmation, not the first opportunity

## Trap Trading Signal Design

### Bullish setup

Signal name:

`FAILED_BEARISH_BIAS_LONG`

#### Step 1: Identify the trap bar

- timeframe: `30m`
- candle is red
- volume is greater than `1.5 x` volume SMA
- store:
  - `trap_high = candle.high`
  - `trap_low = candle.low`

If a full `SMA20` is not available early in the day, use the available rolling intraday window.

#### Step 2: Confirm bearish follow-through failed

Over the next `1-3` candles:

- no close below `trap_low`
- no strong downside range expansion
- futures premium remains healthy, or at least does not weaken materially

This is the filter that separates a real bearish trend bar from an exhaustion trap.

#### Step 3: Trigger long

Trigger when a later candle closes above `trap_high`.

Prefer the setup if at least `2` of the following are also true:

- futures premium `> 0.2%`
- call premium change is positive
- put premium change is negative
- momentum is `UP`
- trigger candle closes in the top `25%` of its range

#### Step 4: Entry and risk

Two entry styles:

- aggressive: enter on close above `trap_high`
- conservative: enter only after trigger candle high breaks

Stop options:

- aggressive: below reclaim candle low
- safer: below `trap_low`

Targets:

- first target: session resistance or prior local swing high
- second target: day high or a range expansion projection
- after first target, trail the stop

### Bearish inverse

Signal name:

`FAILED_BULLISH_BIAS_SHORT`

Rules:

- a green high-volume bias candle forms
- later candles fail to continue higher
- price then closes below that candle's low
- ideally with futures discount, put expansion, and bearish momentum

## Why This Pattern Catches Huge Moves Early

Standard breakout logic is useful for confirmation, but it often enters late on squeeze days.

The earlier edge comes when:

- a strong one-sided candle fails
- that candle's liquidity level gets reclaimed
- derivatives were already leaning in the opposite direction

This combination often appears before the fastest part of the move.

So a good huge-move detector should not only look for:

- breakout
- momentum
- option expansion

It should also look for:

- failed high-volume bias candle
- liquidity reclaim
- derivative divergence in favor of reversal

## External Context

There was no clean same-day negative company-specific catalyst that fully explained the move as a simple bearish continuation event.

The broader context appeared to be:

- early market weakness in the session
- a positive Adani Green capacity-add update around the prior day

So from the tape, the move looked more like:

- selloff absorption
- short trap
- derivative-led squeeze

than a plain news-only breakout.

## Implementation Notes For This Repo

This analysis was the basis for adding the failed-bias style logic into the stock signal workflow.

The repo-friendly implementation pattern is:

- scan each completed candle from session start
- identify high-volume bullish or bearish bias candles
- track whether follow-through fails
- mark the bias candle high or low as the trap liquidity level
- trigger only when a later closed candle reclaims that level
- boost confidence with momentum, futures-vs-VWAP, and option expansion/decay

That keeps the signal focused on the transition from apparent continuation to actual reversal-and-expansion.
