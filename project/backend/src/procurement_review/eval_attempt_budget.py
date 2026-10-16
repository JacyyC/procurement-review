"""Per-attempt exposure accounting; an overrun never cancels an unknown reserve."""

import json
import sqlite3


def token_exposure(connection: sqlite3.Connection) -> tuple[int, int]:
    totals = [0, 0]
    for row in connection.execute(
        "SELECT reserved_input_tokens, reserved_output_tokens, receipt_json FROM attempts"
    ):
        receipt = json.loads(row[2]) if row[2] else {}
        for index, key in enumerate(("input_tokens", "output_tokens")):
            actual = receipt.get(key)
            totals[index] += max(row[index], actual if actual is not None else 0)
    return totals[0], totals[1]
