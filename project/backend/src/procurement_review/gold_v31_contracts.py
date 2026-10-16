"""Parallel corpus/truth revision; the original v3 files and loaders stay frozen."""

from typing import Literal
from pydantic import Field
from .gold_v3_contracts import (
    EvidenceDocument,
    GoldCaseV3,
    GoldCasesV3File,
    GoldManifestV3,
    GoldTruthV1File,
    TruthRecordV1,
    V3ModelInput,
)


class EvidenceDocumentV31(EvidenceDocument):
    version: Literal["synthetic-gold-document.v3.1"] = "synthetic-gold-document.v3.1"


class ModelInputV31(V3ModelInput):
    policy_document: EvidenceDocumentV31
    contract_document: EvidenceDocumentV31


class GoldCaseV31(GoldCaseV3):
    gold_set_version: Literal["gold-set.v3.1"] = "gold-set.v3.1"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.1"] = (
        "synthetic-gold-corpus.v3.1"
    )
    case_version: Literal["2.1"] = "2.1"
    model_input: ModelInputV31


class TruthRecordV11(TruthRecordV1):
    truth_contract: Literal["v1-07-truth.v1.1"] = "v1-07-truth.v1.1"


class CasesFileV31(GoldCasesV3File):
    gold_set_version: Literal["gold-set.v3.1"] = "gold-set.v3.1"
    generator_version: Literal["procurement-gold-generator.v3.1"] = (
        "procurement-gold-generator.v3.1"
    )
    cases: tuple[GoldCaseV31, ...] = Field(min_length=180, max_length=180)


class TruthFileV11(GoldTruthV1File):
    schema_version: Literal["v1-07-truth.v1.1"] = "v1-07-truth.v1.1"
    gold_set_version: Literal["gold-set.v3.1"] = "gold-set.v3.1"
    generator_version: Literal["procurement-gold-generator.v3.1"] = (
        "procurement-gold-generator.v3.1"
    )
    truths: tuple[TruthRecordV11, ...] = Field(min_length=180, max_length=180)


class ManifestV31(GoldManifestV3):
    gold_set_version: Literal["gold-set.v3.1"] = "gold-set.v3.1"
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3.1"] = (
        "synthetic-gold-corpus.v3.1"
    )
    generator_version: Literal["procurement-gold-generator.v3.1"] = (
        "procurement-gold-generator.v3.1"
    )
