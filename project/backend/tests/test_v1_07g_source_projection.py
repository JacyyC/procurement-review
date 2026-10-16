"""Independent public fact identities and settlement transaction counterexamples."""

import copy
from datetime import datetime, timedelta

import pytest

from procurement_review.precise_dev_paired_metering import completeness_report_v3
from test_v1_07g_consumption_v5 import rejected
from v1_07g_consumption_support import fixture_v5, seal


@pytest.fixture(scope="module")
def fact_baseline():
    data = fixture_v5()
    run = next(iter(data[2]["runs"].values()))
    registered = data[0]["truth_extension"][run["case_id"]]
    facts = registered["facts"]
    identity = facts["supplier.registry_status"]
    body = {
        key.split(".")[1]: value["value"]
        for key, value in facts.items()
        if key.startswith("supplier.") and not key.endswith(".identity")
    }
    fact = dict(
        schema_version="1.0",
        tool_name="get_supplier",
        record_id=identity["source_id"],
        dataset_version=identity["version"],
        as_of=registered["as_of"] + "T00:00:00Z",
        status="known",
        result=body,
        failure_reason=None,
        read_only=True,
    )
    decision = next(
        r
        for r in run["execution"]["decision"]["findings"]
        if r["obligation_id"] == "O-03"
    )
    decision["evidence"] = [
        dict(
            source_id=identity["source_id"],
            version=identity["version"],
            locator="supplier.registry_status",
            content_sha256=identity["content_sha256"],
        )
    ]
    summary = run["summary"]["summary_payload"]
    finding = next(r for r in summary["findings"] if r["finding_key"] == "O-03")
    finding.update(evidence_refs=[], business_fact_refs=[fact])
    run["execution"]["summary"] = copy.deepcopy(summary)
    from procurement_review.trace import canonical_checksum

    run["summary"]["summary_checksum"] = canonical_checksum(summary)
    return seal(data)


def test_authored_fact_source_positive(fact_baseline):
    result = completeness_report_v3(*fact_baseline)
    assert result["complete"], result["failures"]


@pytest.mark.parametrize("value", [1, 1.0])
def test_fact_read_only_literal_keeps_its_boolean_type(fact_baseline, value):
    data = copy.deepcopy(fact_baseline)
    run = next(iter(data[2]["runs"].values()))
    summary = run["summary"]["summary_payload"]
    next(r for r in summary["findings"] if r["finding_key"] == "O-03")[
        "business_fact_refs"
    ][0]["read_only"] = value
    run["execution"]["summary"] = copy.deepcopy(summary)
    from procurement_review.trace import canonical_checksum

    run["summary"]["summary_checksum"] = canonical_checksum(summary)
    seal(data)
    rejected(
        data,
        "execution_contract_type:summary.findings.2.business_fact_refs.0.read_only",
    )


def test_public_summary_identity_does_not_equate_integer_with_boolean(fact_baseline):
    data = copy.deepcopy(fact_baseline)
    run = next(iter(data[2]["runs"].values()))
    summary = run["summary"]["summary_payload"]
    next(r for r in summary["findings"] if r["finding_key"] == "O-03")[
        "business_fact_refs"
    ][0]["read_only"] = 1
    from procurement_review.trace import canonical_checksum

    run["summary"]["summary_checksum"] = canonical_checksum(summary)
    seal(data)
    rejected(data, "execution_summary_mismatch")


@pytest.mark.parametrize(
    "field,value",
    [("tool_name", "get_purchase_order"), ("as_of", "2027-06-30T00:00:00Z")],
)
def test_public_fact_metadata_is_bound_to_registered_identity(
    fact_baseline, field, value
):
    data = copy.deepcopy(fact_baseline)
    run = next(iter(data[2]["runs"].values()))
    summary = run["summary"]["summary_payload"]
    next(r for r in summary["findings"] if r["finding_key"] == "O-03")[
        "business_fact_refs"
    ][0][field] = value
    run["execution"]["summary"] = copy.deepcopy(summary)
    from procurement_review.trace import canonical_checksum

    run["summary"]["summary_checksum"] = canonical_checksum(summary)
    seal(data)
    rejected(data, "summary_fact_identity")


def test_nonadopted_settlement_and_receipt_are_one_transaction():
    data = fixture_v5(True)
    run = next(iter(data[2]["runs"].values()))
    run["attempts"][0]["finished_at"] = (
        datetime.fromisoformat(run["created_at"]) + timedelta(milliseconds=500)
    ).isoformat()
    rejected(data, "trace_settlement_transaction")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "source-version-wrong"),
        ("source_kind", "case_document"),
        ("page_number", 2),
        ("section", "lines:99-100:part:1"),
    ],
)
def test_document_reference_keeps_its_ingestion_identity(field, value):
    data = fixture_v5()
    run = next(iter(data[2]["runs"].values()))
    source = run["execution"]["policy"]["evidence_refs"][0]["source_id"]
    refs = [*run["execution"]["policy"]["evidence_refs"]]
    refs.extend(
        ref
        for finding in run["summary"]["summary_payload"]["findings"]
        for ref in finding["evidence_refs"]
        if ref["source_id"] == source
    )
    for ref in refs:
        ref[field] = value
    run["execution"]["summary"] = copy.deepcopy(run["summary"]["summary_payload"])
    run["checkpoints"][0]["result_payload"]["result"]["evidence_refs"] = copy.deepcopy(
        run["execution"]["policy"]["evidence_refs"]
    )
    from procurement_review.trace import canonical_checksum

    run["summary"]["summary_checksum"] = canonical_checksum(
        run["summary"]["summary_payload"]
    )
    seal(data)
    rejected(data, "summary_document_identity")
