
"""
Classe che controlla la validità delle decisioni di rigoletto
"""

import math
import numbers

STATE_DIM = 10
ACTION_COUNT = 3


def validate_state(state):
    values = [float(value) for value in state]

    if len(values) != STATE_DIM:
        raise ValueError(
            f"Stato non valido: attesi {STATE_DIM} valori, "
            f"ricevuti {len(values)}"
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Lo stato contiene valori non finiti")

    return values


def validate_action(action):
    if isinstance(action, (bool,)) or not isinstance(
        action, numbers.Integral
    ):
        raise ValueError(f"Tipo di azione non valido: {action!r}")

    action = int(action)

    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"Azione RIGOLETTO non valida: {action}")

    return action
