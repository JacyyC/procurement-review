"""Read one consistent manifest snapshot, with explicit data/truth version pairing."""

from collections import Counter
from datetime import datetime
from .eval_v2_contracts import BudgetSnapshotV1, EvalRunManifestV2


def data_versions(version):
    truth = {
        "gold-set.v3": "v1-07-truth.v1",
        "gold-set.v3.1": "v1-07-truth.v1.1",
        "gold-set.v3.2": "v1-07-truth.v1.2",
        "gold-set.v3.3": "v1-07-truth.v1.3",
        "gold-set.v3.4": "v1-07-truth.v1.4",
        "gold-set.v3.5": "v1-07-truth.v1.5",
        "gold-set.v3.6": "v1-07-truth.v1.6",
    }[version]
    return {"gold_set_version": version, "truth_contract": truth}


def read_manifest(connection):
    connection.execute("BEGIN")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    versions = data_versions(metadata.get("gold_set_version", "gold-set.v3"))
    if (
        metadata.get("truth_contract", versions["truth_contract"])
        != versions["truth_contract"]
    ):
        raise ValueError("incompatible stored data and truth revision")
    counts = Counter(
        row["state"] for row in connection.execute("SELECT state FROM tasks")
    )
    attempts = connection.execute("SELECT COUNT(*) AS value FROM attempts").fetchone()[
        "value"
    ]
    budget = connection.execute("SELECT * FROM budget WHERE singleton=1").fetchone()
    return EvalRunManifestV2(
        **versions,
        stage_id=metadata.get("stage_id", "V1-07B"),
        run_id=metadata["run_id"],
        gold_manifest_sha256=metadata["gold_manifest_sha256"],
        configuration_sha256=metadata["configuration_sha256"],
        created_at=datetime.fromisoformat(metadata["created_at"]),
        task_count=sum(counts.values()),
        committed_task_count=counts["committed"],
        blocked_unknown_task_count=counts["blocked_unknown"],
        attempt_count=attempts,
        budget=BudgetSnapshotV1(
            **{key: budget[key] for key in BudgetSnapshotV1.model_fields}
        ),
    )
