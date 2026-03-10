from __future__ import annotations

from datetime import datetime
from typing import Dict


def print_signal(payload: Dict[str, object]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 88)
    print(f"[{ts}] Signal: {payload['signal']} | Confidence: {payload.get('confidence')}")
    print(
        f"Bias: {payload['bias']} | Momentum: {payload['momentum']} | "
        f"Support: {payload['support']} | Resistance: {payload['resistance']}"
    )
    print(
        f"Entry: {payload.get('entry')} | Target: {payload.get('target')} | "
        f"Stop: {payload.get('stop_loss')} | RR: {payload.get('rr')}"
    )
    opt = payload.get("option_type")
    if opt:
        print(
            f"Option: {opt} Strike {payload.get('strike_price')} | "
            f"Entry: {payload.get('option_entry')} | SL: {payload.get('option_sl')} | "
            f"Target: {payload.get('option_target')}"
        )
    print(f"Reasons: {'; '.join(payload['score'].get('reasons', []))}")
    print(f"Notes: {'; '.join(payload.get('notes', []))}")
    print("=" * 88)
