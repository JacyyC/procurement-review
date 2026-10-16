"""Crash-safe, exclusive per-observation H commits and shared attempt quota."""

from pathlib import Path

from .eval_h_storage import locked, private, read, write
from .gold_v3_contracts import payload_sha256


def key(case, seed, arm):
    return f"{case}.{seed}.{arm}"


class Journal:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.root = self.directory / "observations"
        self.root.mkdir(mode=0o700, exist_ok=True)
        private(self.root, directory=True)

    def path(self, case, seed, arm):
        return self.root / (key(case, seed, arm) + ".json")

    def committed(self, case, seed, arm):
        path = self.path(case, seed, arm)
        if not path.exists():
            return None
        private(path)
        body = read(path)
        if type(body) is not dict:
            raise ValueError("h_observation_commit_payload")
        if body.get("commit_sha256") != payload_sha256(
            {k: v for k, v in body.items() if k != "commit_sha256"}
        ):
            raise ValueError("h_observation_commit_checksum")
        return body

    def commit(self, body):
        body = {**body, "commit_sha256": payload_sha256(body)}
        path = self.path(body["case_id"], body["repeat_seed"], body["arm"])
        if path.exists():
            if payload_sha256(read(path)) != payload_sha256(body):
                raise ValueError("h_observation_already_committed")
        else:
            write(path, body)
        return body

    def task_directory(self, case, seed, arm):
        parent = self.directory / "attempts"
        parent.mkdir(mode=0o700, exist_ok=True)
        private(parent, directory=True)
        root = parent / key(case, seed, arm)
        root.mkdir(mode=0o700, exist_ok=True)
        private(root, directory=True)
        return root

    def reserve(self, case, seed, arm, branch, request_sha256):
        root = self.task_directory(case, seed, arm)
        with locked(root):
            old = sorted(root.glob(branch + ".*.json"))
            if len(old) >= 2:
                raise ValueError("h_shared_attempt_budget_exhausted")
            for path in old:
                if read(path)["status"] == "started":
                    raise ValueError("h_unsettled_attempt_no_resend")
            number = len(old) + 1
            path = root / f"{branch}.{number}.json"
            write(
                path,
                dict(
                    case_id=case,
                    repeat_seed=seed,
                    arm=arm,
                    branch=branch,
                    attempt_no=number,
                    request_sha256=request_sha256,
                    status="started",
                ),
            )
            return path

    def settle(self, path, receipt, *, projection=None):
        with locked(path.parent):
            private(path)
            body = read(path)
            if body["status"] != "started":
                raise ValueError("h_attempt_already_settled")
            write(
                path,
                {
                    **body,
                    "status": "settled",
                    "receipt": receipt,
                    "projection": projection,
                    "projection_sha256": payload_sha256(projection),
                },
                replace=True,
            )

    def recover_unknown(self, case, seed, arm):
        """No receipt is never a success. Recovery keeps the sent slot consumed."""
        for path in self.task_directory(case, seed, arm).glob("*.json"):
            if read(path)["status"] == "started":
                self.settle(
                    path,
                    dict(
                        error_code="interrupted_unknown",
                        usage_unknown=True,
                        input_tokens=None,
                        output_tokens=None,
                        returned=False,
                        latency_ms=None,
                        resolved_model=None,
                    ),
                    projection=None,
                )

    def attempts(self, case, seed, arm):
        return [
            read(path)
            for path in sorted(self.task_directory(case, seed, arm).glob("*.json"))
        ]
