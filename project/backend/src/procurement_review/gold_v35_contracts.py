"""Fresh validation revision; prior corpora and truth files remain immutable."""

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


class EvidenceDocumentV35(EvidenceDocumentV31):
    version: Literal["synthetic-gold-document.v3.5"] = "synthetic-gold-document.v3.5"


class ModelInputV35(ModelInputV31):
    policy_document: EvidenceDocumentV35
    contract_document: EvidenceDocumentV35


class GoldCaseV35(GoldCaseV31):
    gold_set_version: Literal["gold-set.v3.5"] = "gold-set.v3.5"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.5"] = (
        "synthetic-gold-corpus.v3.5"
    )
    case_version: Literal["2.5"] = "2.5"
    model_input: ModelInputV35


class TruthRecordV15(TruthRecordV11):
    truth_contract: Literal["v1-07-truth.v1.5"] = "v1-07-truth.v1.5"


class CasesFileV35(CasesFileV31):
    seed: Literal[20260923] = 20260923
    gold_set_version: Literal["gold-set.v3.5"] = "gold-set.v3.5"
    generator_version: Literal["procurement-gold-generator.v3.5"] = (
        "procurement-gold-generator.v3.5"
    )
    cases: tuple[GoldCaseV35, ...] = Field(min_length=180, max_length=180)


class TruthFileV15(TruthFileV11):
    seed: Literal[20260923] = 20260923
    schema_version: Literal["v1-07-truth.v1.5"] = "v1-07-truth.v1.5"
    gold_set_version: Literal["gold-set.v3.5"] = "gold-set.v3.5"
    generator_version: Literal["procurement-gold-generator.v3.5"] = (
        "procurement-gold-generator.v3.5"
    )
    truths: tuple[TruthRecordV15, ...] = Field(min_length=180, max_length=180)


class ManifestV35(ManifestV31):
    seed: Literal[20260923] = 20260923
    gold_set_version: Literal["gold-set.v3.5"] = "gold-set.v3.5"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.5"] = (
        "synthetic-gold-corpus.v3.5"
    )
    generator_version: Literal["procurement-gold-generator.v3.5"] = (
        "procurement-gold-generator.v3.5"
    )
