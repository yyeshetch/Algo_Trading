# IV, IVP & Option Parameters for Option Buying

> Are IV, IVP, and other option parameters important for option buying?

**Yes.** For option **buying** (long premium), volatility metrics are arguably more
important than for any other style, because as a buyer you are **long vega** and
**short theta**. You can be right on direction and still lose money if you buy when
volatility is rich. Below is how each parameter matters, ranked by practical
relevance for an intraday / short-term index option buyer.

---

## 1. The ones that matter most

### IV (Implied Volatility)
- The market's expected volatility priced into the option.
- As a buyer, **high IV = expensive premium** → you need a bigger/faster move to
  profit. **Low IV = cheaper, more favorable entry.**
- The real killer is **IV crush**: buying into an event (results, RBI policy,
  budget, expiry) when IV is elevated, then IV collapses after the event and your
  option loses value **even if direction is right**.

### IVP (IV Percentile) / IVR (IV Rank)
- IV in isolation is meaningless — 15% IV is high for one underlying and low for
  another. **IVP tells you where current IV sits vs its own past** (e.g. last 1 year).
- Rule of thumb:
  - **IVP low (< 30–40) → favors buying** (volatility cheap, room to expand).
  - **IVP high (> 70) → favors selling / caution for buyers.**
- For intraday index buying, the **IV trend today** (is IV rising or falling
  intraday) matters more than the 1-year percentile, but the percentile sets the
  backdrop.

---

## 2. The Greeks (day-to-day P&L drivers)

| Greek | What it is | Why it matters to a buyer |
|-------|-----------|---------------------------|
| **Theta** | Time decay | Your biggest enemy. Decays fastest on expiry day and into the close. ATM bleeds most. Holding a loser "hoping" costs theta every minute. |
| **Delta** | Directional exposure / probability proxy | ATM ≈ 0.5. Slightly ITM (0.55–0.65) for trend rides (less theta/IV dependence); OTM for cheap lottery on a fast move. |
| **Gamma** | How fast delta changes | High near ATM and near expiry. Explosive gains on a move (good) but fast losses too. Expiry-day ATM is a pure gamma/theta fight. |
| **Vega** | Sensitivity to IV change | You are long vega → you *want* IV to rise after entry. Buying in low IV and getting an IV expansion (breakout sparks panic) is the ideal buyer setup. |

---

## 3. Structural / flow parameters

- **Premium vs OI together** — rising premium + rising OI = **fresh buying** (real);
  rising premium + falling OI = **short covering**. Cleanest "is the move real" filter.
- **ATM straddle price / expected move** — the ATM CE+PE premium implies the day's
  expected range. If your target is smaller than the straddle-implied move, the trade
  is statistically poor for a buyer.
- **PCR, OI skew** — context for positioning.

---

## 4. Bottom line — the ideal long-premium setup

1. **Direction confirmed** (spot/VWAP/futures/flow confluence).
2. **IV not elevated** (low-to-mid IVP, ideally IV starting to tick up).
3. **Not buying into known IV crush** (avoid pre-event ATM longs unless that's the
   explicit thesis).
4. **Enough expected move** (target > a meaningful fraction of the straddle / ATM premium).
5. **Theta-aware exit discipline** (especially expiry day).

**Summary:** IV and IVP are not optional polish — they decide whether the option is
cheap or expensive, which directly sets your win-rate as a buyer. Direction tells you
*which way*; IV / IVP / Greeks tell you *whether the option is worth buying at all and
how fast it will move/decay*.

---

## 5. Proposed dashboard enhancement (option-buyer volatility panel)

Extend the Analysis / confluence tab with an option-buyer volatility panel:

- Live **ATM IV**
- **Intraday IV trend** (rising / falling)
- **IV-percentile gauge** (IVP / IVR)
- **ATM straddle expected-move vs your target**
- **Theta / gamma warning** near expiry

This turns the points above into concrete checks alongside the directional confluence
already in place.

**Implementation note:** Kite's option quotes don't expose IV directly, so IV would be
computed via **Black–Scholes** from LTP + spot + time-to-expiry, and **IVP** from a
rolling history we'd start storing.
