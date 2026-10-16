"""Parallel replacement validation corpus. Original v3.1 remains immutable."""

from typing import Literal
from pydantic import Field
from .gold_v31_contracts import (
    EvidenceDocumentV31,
    ModelInputV31,
    GoldCaseV31,
    TruthRecordV11,
    CasesFileV31,
    TruthFileV11,
    ManifestV31,
)


class EvidenceDocumentV32(EvidenceDocumentV31):
    version: Literal["synthetic-gold-document.v3.2"] = "synthetic-gold-document.v3.2"


class ModelInputV32(ModelInputV31):
    policy_document: EvidenceDocumentV32
    contract_document: EvidenceDocumentV32


class GoldCaseV32(GoldCaseV31):
    gold_set_version: Literal["gold-set.v3.2"] = "gold-set.v3.2"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.2"] = (
        "synthetic-gold-corpus.v3.2"
    )
    case_version: Literal["2.2"] = "2.2"
    model_input: ModelInputV32


class TruthRecordV12(TruthRecordV11):
    truth_contract: Literal["v1-07-truth.v1.2"] = "v1-07-truth.v1.2"


class CasesFileV32(CasesFileV31):
    seed: Literal[20260920] = 20260920
    gold_set_version: Literal["gold-set.v3.2"] = "gold-set.v3.2"
    generator_version: Literal["procurement-gold-generator.v3.2"] = (
        "procurement-gold-generator.v3.2"
    )
    cases: tuple[GoldCaseV32, ...] = Field(min_length=180, max_length=180)


class TruthFileV12(TruthFileV11):
    seed: Literal[20260920] = 20260920
    schema_version: Literal["v1-07-truth.v1.2"] = "v1-07-truth.v1.2"
    gold_set_version: Literal["gold-set.v3.2"] = "gold-set.v3.2"
    generator_version: Literal["procurement-gold-generator.v3.2"] = (
        "procurement-gold-generator.v3.2"
    )
    truths: tuple[TruthRecordV12, ...] = Field(min_length=180, max_length=180)


class ManifestV32(ManifestV31):
    seed: Literal[20260920] = 20260920
    gold_set_version: Literal["gold-set.v3.2"] = "gold-set.v3.2"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.2"] = (
        "synthetic-gold-corpus.v3.2"
    )
    generator_version: Literal["procurement-gold-generator.v3.2"] = (
        "procurement-gold-generator.v3.2"
    )
