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


class EvidenceDocumentV36(EvidenceDocumentV31):
    version: Literal["synthetic-gold-document.v3.6"] = "synthetic-gold-document.v3.6"


class ModelInputV36(ModelInputV31):
    policy_document: EvidenceDocumentV36
    contract_document: EvidenceDocumentV36


class GoldCaseV36(GoldCaseV31):
    gold_set_version: Literal["gold-set.v3.6"] = "gold-set.v3.6"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.6"] = (
        "synthetic-gold-corpus.v3.6"
    )
    case_version: Literal["2.6"] = "2.6"
    model_input: ModelInputV36


class TruthRecordV16(TruthRecordV11):
    truth_contract: Literal["v1-07-truth.v1.6"] = "v1-07-truth.v1.6"


class CasesFileV36(CasesFileV31):
    seed: Literal[20260924] = 20260924
    gold_set_version: Literal["gold-set.v3.6"] = "gold-set.v3.6"
    generator_version: Literal["procurement-gold-generator.v3.6"] = (
        "procurement-gold-generator.v3.6"
    )
    cases: tuple[GoldCaseV36, ...] = Field(min_length=180, max_length=180)


class TruthFileV16(TruthFileV11):
    seed: Literal[20260924] = 20260924
    schema_version: Literal["v1-07-truth.v1.6"] = "v1-07-truth.v1.6"
    gold_set_version: Literal["gold-set.v3.6"] = "gold-set.v3.6"
    generator_version: Literal["procurement-gold-generator.v3.6"] = (
        "procurement-gold-generator.v3.6"
    )
    truths: tuple[TruthRecordV16, ...] = Field(min_length=180, max_length=180)


class ManifestV36(ManifestV31):
    seed: Literal[20260924] = 20260924
    gold_set_version: Literal["gold-set.v3.6"] = "gold-set.v3.6"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.6"] = (
        "synthetic-gold-corpus.v3.6"
    )
    generator_version: Literal["procurement-gold-generator.v3.6"] = (
        "procurement-gold-generator.v3.6"
    )
