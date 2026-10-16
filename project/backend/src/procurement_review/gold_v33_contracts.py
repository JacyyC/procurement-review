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


class EvidenceDocumentV33(EvidenceDocumentV31):
    version: Literal["synthetic-gold-document.v3.3"] = "synthetic-gold-document.v3.3"


class ModelInputV33(ModelInputV31):
    policy_document: EvidenceDocumentV33
    contract_document: EvidenceDocumentV33


class GoldCaseV33(GoldCaseV31):
    gold_set_version: Literal["gold-set.v3.3"] = "gold-set.v3.3"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.3"] = (
        "synthetic-gold-corpus.v3.3"
    )
    case_version: Literal["2.3"] = "2.3"
    model_input: ModelInputV33


class TruthRecordV13(TruthRecordV11):
    truth_contract: Literal["v1-07-truth.v1.3"] = "v1-07-truth.v1.3"


class CasesFileV33(CasesFileV31):
    seed: Literal[20260921] = 20260921
    gold_set_version: Literal["gold-set.v3.3"] = "gold-set.v3.3"
    generator_version: Literal["procurement-gold-generator.v3.3"] = (
        "procurement-gold-generator.v3.3"
    )
    cases: tuple[GoldCaseV33, ...] = Field(min_length=180, max_length=180)


class TruthFileV13(TruthFileV11):
    seed: Literal[20260921] = 20260921
    schema_version: Literal["v1-07-truth.v1.3"] = "v1-07-truth.v1.3"
    gold_set_version: Literal["gold-set.v3.3"] = "gold-set.v3.3"
    generator_version: Literal["procurement-gold-generator.v3.3"] = (
        "procurement-gold-generator.v3.3"
    )
    truths: tuple[TruthRecordV13, ...] = Field(min_length=180, max_length=180)


class ManifestV33(ManifestV31):
    seed: Literal[20260921] = 20260921
    gold_set_version: Literal["gold-set.v3.3"] = "gold-set.v3.3"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.3"] = (
        "synthetic-gold-corpus.v3.3"
    )
    generator_version: Literal["procurement-gold-generator.v3.3"] = (
        "procurement-gold-generator.v3.3"
    )
