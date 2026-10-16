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


class EvidenceDocumentV34(EvidenceDocumentV31):
    version: Literal["synthetic-gold-document.v3.4"] = "synthetic-gold-document.v3.4"


class ModelInputV34(ModelInputV31):
    policy_document: EvidenceDocumentV34
    contract_document: EvidenceDocumentV34


class GoldCaseV34(GoldCaseV31):
    gold_set_version: Literal["gold-set.v3.4"] = "gold-set.v3.4"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.4"] = (
        "synthetic-gold-corpus.v3.4"
    )
    case_version: Literal["2.4"] = "2.4"
    model_input: ModelInputV34


class TruthRecordV14(TruthRecordV11):
    truth_contract: Literal["v1-07-truth.v1.4"] = "v1-07-truth.v1.4"


class CasesFileV34(CasesFileV31):
    seed: Literal[20260922] = 20260922
    gold_set_version: Literal["gold-set.v3.4"] = "gold-set.v3.4"
    generator_version: Literal["procurement-gold-generator.v3.4"] = (
        "procurement-gold-generator.v3.4"
    )
    cases: tuple[GoldCaseV34, ...] = Field(min_length=180, max_length=180)


class TruthFileV14(TruthFileV11):
    seed: Literal[20260922] = 20260922
    schema_version: Literal["v1-07-truth.v1.4"] = "v1-07-truth.v1.4"
    gold_set_version: Literal["gold-set.v3.4"] = "gold-set.v3.4"
    generator_version: Literal["procurement-gold-generator.v3.4"] = (
        "procurement-gold-generator.v3.4"
    )
    truths: tuple[TruthRecordV14, ...] = Field(min_length=180, max_length=180)


class ManifestV34(ManifestV31):
    seed: Literal[20260922] = 20260922
    gold_set_version: Literal["gold-set.v3.4"] = "gold-set.v3.4"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.4"] = (
        "synthetic-gold-corpus.v3.4"
    )
    generator_version: Literal["procurement-gold-generator.v3.4"] = (
        "procurement-gold-generator.v3.4"
    )
