# High-Move Strategies

This document lists `10` core strategies plus additional trap-trading setups designed to capture large intraday or swing moves.

These are not random chart patterns. Each setup is built around one of the following ideas:

- trapped positioning
- trend continuation after compression
- breakout with derivative confirmation
- momentum expansion after structure reclaim

The goal is to focus on moves that can travel far enough to justify execution risk.

Trap trading gets special emphasis because some of the largest moves do not begin from clean continuation. They begin when:

- a strong directional candle fails
- trapped participants cannot get follow-through
- liquidity gets reclaimed
- derivatives start confirming the reversal before price looks obvious

## Working Definition

For this document, a strategy is considered useful only if it can satisfy most of these conditions:

- it has a clear structural trigger
- it can be defined using objective rules
- it has a logical invalidation point
- it tends to appear before or during a large directional move
- it can be implemented in code using OHLCV, futures, and options context where available

## 1. Failed Bearish Bias Reclaim

### Use case

Intraday. Best on `15m` or `30m` for stocks and indices.

### Core idea

A heavy red candle appears, looks bearish, but price fails to continue lower. When that candle high gets reclaimed, trapped shorts fuel the upside move.

### Setup

- identify a red candle
- volume is greater than `1.5 x` rolling volume SMA
- store:
  - `trap_high`
  - `trap_low`

### Trigger

- next `1-3` candles fail to close below `trap_low`
- a later candle closes above `trap_high`

### Best confirmations

- futures premium stays positive
- calls expand
- puts decay
- momentum flips `UP`
- trigger candle closes near its high

### Stop

- below reclaim candle low
- safer stop below `trap_low`

### Target

- prior swing high
- session resistance
- range expansion target

### Why it captures huge moves

The move is powered by failed bearish continuation plus short covering.

## 2. Failed Bullish Bias Breakdown

### Use case

Intraday. Best on `15m` or `30m`.

### Core idea

A strong green candle traps longs, but follow-through fails. Once price closes below that candle low, bullish positioning starts unwinding quickly.

### Setup

- identify a green candle
- volume is greater than `1.5 x` rolling volume SMA
- store:
  - `trap_high`
  - `trap_low`

### Trigger

- next `1-3` candles fail to close above `trap_high`
- later candle closes below `trap_low`

### Best confirmations

- futures weaken below VWAP
- puts expand
- calls decay
- momentum flips `DOWN`
- trigger candle closes near its low

### Stop

- above reclaim candle high
- safer stop above `trap_high`

### Target

- prior swing low
- session support
- measured breakdown projection

### Why it captures huge moves

It combines failed bullish continuation with long liquidation.

## 3. Opening Range Breakout With Retest Hold

### Use case

Intraday. Best for trend days in high-beta F&O names.

### Core idea

The stock breaks the opening range, retests the breakout level, and then resumes in the breakout direction. This avoids chasing the first spike and often catches the cleanest expansion leg.

### Setup

- define opening range from first `15m` candle
- price closes above OR high for bullish setup or below OR low for bearish setup

### Trigger

- price retests OR level
- retest holds without deep rejection
- next candle resumes in breakout direction

### Best confirmations

- futures stay stronger than spot on bullish setup
- options support direction:
  - bullish: calls expand, puts decay
  - bearish: puts expand, calls decay
- breakout happens with rising volume

### Stop

- just beyond OR level
- or beyond retest candle extreme

### Target

- `1.5R`
- `2R`
- trail on momentum candles

### Why it captures huge moves

True trend days often defend ORB levels and trend cleanly after the retest.

## 4. Volume Pocket Reclaim

### Use case

Intraday and short swing. Works well after news-driven spikes or dumps.

### Core idea

A high-volume candle defines a key pocket of acceptance. If price later reclaims that pocket after briefly losing it, the move often accelerates because trapped traders are forced to reverse.

### Setup

- mark the high, low, and midpoint of the highest-volume candle of the session or recent swing
- wait for price to move away and then test that area again

### Trigger

- bullish: price reclaims the midpoint and closes above the candle high
- bearish: price loses the midpoint and closes below the candle low

### Best confirmations

- reclaim occurs with improving breadth or futures structure
- reversal candle has above-average volume
- options rotate with the reclaim

### Stop

- other side of the volume pocket

### Target

- prior impulse high or low
- day high or low
- next major volume node

### Why it captures huge moves

Large moves often start where the market shifts from rejection to acceptance around volume-heavy zones.

## 5. VWAP Trend Day Continuation

### Use case

Intraday. Best on strong institutional trend days.

### Core idea

Once a stock establishes trend above or below VWAP and keeps defending it, pullbacks to VWAP become continuation entries rather than reversal attempts.

### Setup

- bullish:
  - price above session VWAP
  - futures above VWAP
  - higher lows continue forming
- bearish:
  - price below session VWAP
  - futures below VWAP
  - lower highs continue forming

### Trigger

- first or second pullback into VWAP
- rejection candle forms in trend direction
- next candle confirms continuation

### Best confirmations

- directional option premium behavior
- rising OI in the move direction
- no strong opposing liquidity grab

### Stop

- below pullback low for bullish
- above pullback high for bearish

### Target

- intraday measured move
- prior extension leg projected from VWAP
- trail behind each new swing

### Why it captures huge moves

Trend days can produce multiple expansion legs, and VWAP often acts as the institutional defense line.

## 6. Volatility Compression Expansion

### Use case

Intraday and swing. Best after a tight range or multi-candle squeeze.

### Core idea

When range and realized volatility compress hard, the next clean expansion often becomes the largest move of the session or week.

### Setup

- `3-8` candles of contracting range
- volume dries up during compression
- highs and lows tighten into a box or triangle

### Trigger

- breakout candle closes outside the compression range
- breakout candle has expanding volume

### Best confirmations

- futures direction matches breakout
- options reprice quickly in the breakout direction
- breakout holds on the very next candle

### Stop

- opposite side of compression range

### Target

- height of the compression box
- prior major swing
- runner with trailing stop

### Why it captures huge moves

Energy stored during compression often releases in a fast directional burst.

## 7. Gap Trap Reversal

### Use case

Intraday. Very effective after emotional gaps.

### Core idea

The market gaps sharply, attracts momentum traders in the gap direction, then fails to continue. Once the opening imbalance gets rejected, the reversal can become violent.

### Setup

- significant gap up or gap down relative to previous close
- opening candle extends gap direction but fails to sustain

### Trigger

- gap up trap:
  - price loses opening range low
  - futures weaken
  - puts expand
- gap down trap:
  - price reclaims opening range high
  - futures strengthen
  - calls expand

### Best confirmations

- large opening wick
- volume spike with poor follow-through
- reclaim of previous day close

### Stop

- above trap high for bearish
- below trap low for bullish

### Target

- previous day close
- gap fill
- opposite session extreme

### Why it captures huge moves

Gap traps force the earliest participants to unwind quickly, creating fuel for a large reversal.

## 8. Higher Timeframe Pullback Continuation

### Use case

Swing. Works on `daily` with `weekly` context.

### Core idea

A strong stock pulls back into a major support zone without changing the higher-timeframe trend. When momentum resumes, the next leg can be large and relatively clean.

### Setup

- weekly trend is up for bullish or down for bearish
- daily pullback retraces into:
  - moving average
  - prior breakout zone
  - anchored VWAP
  - weekly support or resistance

### Trigger

- reversal candle forms at the pullback zone
- next candle confirms in trend direction

### Best confirmations

- sector remains strong in same direction
- futures or volume support resumption
- no major higher-timeframe rejection overhead

### Stop

- beyond pullback swing low or high

### Target

- prior swing high or low
- weekly expansion target
- trend-based trailing stop

### Why it captures huge moves

Large swing advances are often built from pullback-resume cycles, not only from fresh breakouts.

## 9. Base Breakout With Derivative Confirmation

### Use case

Swing and positional. Works well in stocks coming out of multi-week consolidation.

### Core idea

A long base forms, weak hands get cleared out, and then price breaks above the base with derivative support. This often starts a large directional trend.

### Setup

- price consolidates for multiple sessions or weeks
- base highs and lows become well-defined
- volatility contracts into the breakout

### Trigger

- close above base high for bullish
- close below base low for bearish

### Best confirmations

- breakout volume above average
- futures lead spot
- option premiums reprice in breakout direction
- breakout does not instantly fail next session

### Stop

- inside the base
- or beyond breakout candle low or high

### Target

- measured move equal to base height
- next weekly resistance or support
- trail with swing structure

### Why it captures huge moves

The longer the base, the more powerful the breakout can become if participation expands at the break.

## 10. Relative Strength Leader Expansion

### Use case

Intraday and swing. Best for leaders during strong sector rotation.

### Core idea

When one stock refuses to weaken with the market or outperforms its sector consistently, the eventual breakout often travels farther than average because it is already under accumulation.

### Setup

- stock holds above VWAP or key support while the index or sector is flat or weak
- bullish relative strength persists across multiple candles or days

### Trigger

- stock breaks local resistance while broader market stabilizes
- or stock reclaims a pullback level and resumes

### Best confirmations

- sector leaders move together
- futures remain premium to spot
- calls expand faster than peers
- volume increases on the breakout

### Stop

- under relative-strength pivot low

### Target

- next daily or weekly resistance
- measured move based on prior impulse
- runner with trend trail

### Why it captures huge moves

The strongest names often move the farthest once the market gives them room.

## Additional Trap Trading Setups

The first `10` strategies already include some trap logic, but the following setups are more explicitly built around failure, reclaim, and forced positioning unwind.

## 11. Opening Drive Failure

### Use case

Intraday. Best in the first `30-90` minutes.

### Core idea

The opening move looks powerful and attracts early breakout traders, but the market cannot extend it. Once the opening drive fails, the reversal often becomes the strongest move of the morning.

### Setup

- first `1-3` candles form an aggressive directional drive
- volume is above average
- price extends away from previous close quickly

### Trigger

- bullish failure: strong opening rally loses its base and closes back below the opening drive low
- bearish failure: strong opening flush reclaims the opening drive high and closes back above it

### Best confirmations

- large wick on the opening impulse
- futures stop confirming the opening direction
- option premiums flip against the opening move
- reclaim candle closes in the top or bottom `25%` of its range

### Stop

- beyond the extreme of the failed opening drive

### Target

- previous close
- opposite side of opening range
- session trend expansion if reversal becomes a trend day

### Why it captures huge moves

The opening drive traps the most aggressive traders. Once it fails, the unwind can be fast and one-directional.

## 12. Previous Day High/Low Sweep Reversal

### Use case

Intraday and short swing. Very useful around obvious external liquidity levels.

### Core idea

Price sweeps the previous day high or low, attracts breakout participation, and then sharply rejects the level. The best trades happen when price closes back inside the prior day range.

### Setup

- mark `PDH` and `PDL`
- wait for price to trade beyond one of those levels
- sweep should occur with momentum or visible stop run behavior

### Trigger

- bullish: price sweeps `PDL` and then closes back above it
- bearish: price sweeps `PDH` and then closes back below it

### Best confirmations

- sweep candle has long rejection wick
- follow-through beyond the swept level is weak
- futures and options diverge from the apparent breakout
- reclaim happens within `1-2` candles

### Stop

- beyond the sweep extreme

### Target

- midpoint of previous day range
- opposite side of previous day range
- session expansion if the reversal develops into trend

### Why it captures huge moves

Previous day levels hold a lot of liquidity. Failed breaks there often become powerful reversals.

## 13. VWAP Flush Reclaim

### Use case

Intraday. Best on days where the broader trend is intact but price momentarily loses VWAP.

### Core idea

Price briefly flushes below VWAP in an uptrend or spikes above VWAP in a downtrend, pulling in countertrend traders. When VWAP gets reclaimed quickly, the original direction often resumes with force.

### Setup

- bullish version:
  - prior structure is above VWAP
  - pullback breaks below VWAP briefly
- bearish version:
  - prior structure is below VWAP
  - bounce breaks above VWAP briefly

### Trigger

- bullish: reclaim of VWAP plus close back above the reclaim candle high
- bearish: loss of VWAP plus close back below the rejection candle low

### Best confirmations

- trend structure remains intact on higher lows or lower highs
- futures remain aligned with the broader direction
- options flip back in favor of the original trend
- reclaim candle has above-average volume

### Stop

- beyond the VWAP flush extreme

### Target

- trend leg extension
- prior session high or low
- measured move from VWAP reclaim point

### Why it captures huge moves

VWAP flushes often trap countertrend traders right before the strongest continuation leg resumes.

## 14. Breakout Failure To Base Opposite

### Use case

Intraday and swing. Strong when a well-watched breakout level fails immediately.

### Core idea

A breakout occurs from a visible base, but price cannot hold outside the breakout zone. When it closes back inside the base, the failed breakout often rotates to the opposite side of the range with force.

### Setup

- visible multi-candle base
- breakout above resistance or breakdown below support
- breakout is obvious enough to attract participation

### Trigger

- price closes back inside the original base
- next candle confirms movement toward the opposite side of the range

### Best confirmations

- breakout volume was high but follow-through was weak
- derivatives stop confirming the breakout direction
- reversal candle closes decisively inside the prior range

### Stop

- beyond failed breakout extreme

### Target

- midpoint of the base
- opposite side of the base
- expanded move if the opposite-side break also gives way

### Why it captures huge moves

Failed breakouts often become fast range traversals because breakout traders are trapped in the worst location.

## 15. News Spike Absorption Reversal

### Use case

Intraday and swing. Best after headline-driven emotional moves.

### Core idea

News creates a violent move, but instead of trending cleanly, the spike gets absorbed. If price reclaims the spike origin or the spike liquidity level, the reversal can become more powerful than the initial news move.

### Setup

- sudden large candle or gap caused by a visible event
- volume and volatility spike sharply
- initial move stretches away from recent value

### Trigger

- bearish news spike fails and price reclaims the spike high
- bullish news spike fails and price loses the spike low

### Best confirmations

- next candles fail to extend the news move
- derivatives rotate opposite the headline direction
- price retakes the spike midpoint first, then the key extreme
- futures premium or discount diverges from spot panic

### Stop

- beyond the news spike extreme

### Target

- pre-spike origin
- prior structural level
- full squeeze unwind if move becomes crowded

### Why it captures huge moves

Headline moves attract the most emotional participation. If the market absorbs that flow, the reversal can be large because the initial move was crowded and poorly positioned.

## 16. Inside Bar False Break Expansion

### Use case

Intraday and swing. Useful after temporary balance before expansion.

### Core idea

An inside bar or tight inside cluster forms, price briefly breaks one side, then immediately reverses and expands through the opposite side. The false break clears one side of liquidity before the real move starts.

### Setup

- one or more inside bars form after an impulse or pause
- the range becomes obvious to short-term traders

### Trigger

- price breaks one side of the inside bar range
- fails to continue
- then closes through the opposite side

### Best confirmations

- false break occurs on weak follow-through
- true break occurs on stronger close and better volume
- futures and options confirm only the second break

### Stop

- beyond the false-break extreme

### Target

- measured move using inside range height
- prior major swing
- session trend extension

### Why it captures huge moves

The market often clears one side of a tight range first, then launches the real directional move after liquidity is taken.

## Ranking For This Repo

If the goal is to capture large moves using the current repo architecture, the most implementation-ready strategies are:

1. `Failed Bearish Bias Reclaim`
2. `Failed Bullish Bias Breakdown`
3. `Opening Range Breakout With Retest Hold`
4. `VWAP Trend Day Continuation`
5. `Volume Pocket Reclaim`
6. `Volatility Compression Expansion`
7. `Opening Drive Failure`
8. `Previous Day High/Low Sweep Reversal`
9. `VWAP Flush Reclaim`
10. `Inside Bar False Break Expansion`

These are the easiest to define with completed candles, volume, futures, and option premium behavior.

## Suggested Data Inputs

To make these strategies robust in code, the most useful inputs are:

- spot OHLCV
- futures OHLCV
- futures premium vs spot
- session VWAP
- rolling volume SMA
- call and put premium change
- call and put OI change
- opening range
- previous day high, low, close
- swing highs and lows

## Final Note

The best large-move strategies usually do not come from prediction alone. They come from recognizing:

- when continuation fails
- when positioning gets trapped
- when derivatives disagree with price
- when compression turns into expansion

That is where the biggest intraday and swing opportunities usually begin.
