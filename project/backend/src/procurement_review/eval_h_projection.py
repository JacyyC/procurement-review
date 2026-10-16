"""H retains full native evidence identities instead of the legacy truncated summary."""

from .eval_f_lineage import execution_projection_v3


def project(row, binding):
    result = execution_projection_v3(row, binding)
    result["pipeline_failures"] = list(
        binding["execution"].get("decision", {}).get("pipeline_failures", [])
    )
    native = {
        finding["obligation_id"]: finding
        for finding in binding["execution"].get("decision", {}).get("findings", [])
    }
    for finding in result["findings"]:
        finding["evidence"] = native[finding["obligation_id"]]["evidence"]
    return result
