"""Materialize a parallel, evidence-complete DEV/validation revision; no blind."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .gold_v3 import build_gold_v3, validate_gold_v3, _atomic_write
from .gold_v3_contracts import V3Split, payload_sha256
from .gold_v31_contracts import (
    CasesFileV31,
    EvidenceDocumentV31,
    GoldCaseV31,
    ManifestV31,
    ModelInputV31,
    TruthFileV11,
)
from .gold_v31_oracle import derive_truth

DIRECTORY = Path(__file__).parent / "resources/gold_set/v3.1"


def _document(original, content):
    return EvidenceDocumentV31(
        source_id=original.source_id,
        section=original.section,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def revised_case(case) -> GoldCaseV31:
    facts = case.model_input.business_facts
    old = case.model_input.contract_document
    # New synthetic evidence, not a rewrite of old files or a model-output repair.
    if case.split is V3Split.DEV:
        subject = old.content.split("合同摘录", 1)[0].removeprefix("《")
        added = (
            f"\n买方 {facts.counterparty_reference}；卖方 {facts.supplier_id}。标的：{subject}。\n"
            "验收标准：到货数量逐项相符且完好率100%；质保责任：卖方免费修复或更换缺陷交付物。\n"
            "变更程序：双方书面签署后变更生效；违约责任：逾期交付每天按订单价款0.1%赔偿，累计赔偿责任上限为 100000.00 CNY。\n"
            "本例仅适用上述变更、违约和责任上限事项，不增加分包、知识产权或争议程序检查。"
        )
        content = (
            old.content.replace("责任上限条款已列明", "具体责任上限见下文") + added
        )
    else:
        added = (
            f"\nBUYER={facts.counterparty_reference}|SELLER={facts.supplier_id}|SUBJECT=deliverables in the referenced order\n"
            "ACCEPTANCE=all quantities match and 100% undamaged|WARRANTY=seller repairs or replaces defective deliverables at no charge\n"
            "CHANGE=effective after both parties sign in writing|LATE_DELIVERY_DAMAGES=0.1% of order price per day|LIABILITY_LIMIT=100000.00 CNY\n"
            "APPLICABLE_RIGHTS=written change, late delivery damages, liability limit only; no additional subcontracting, IP or dispute checks"
        )
        content = old.content + added
    model_input = ModelInputV31(
        question=case.model_input.question,
        policy_document=_document(
            case.model_input.policy_document, case.model_input.policy_document.content
        ),
        contract_document=_document(old, content),
        business_facts=facts,
    )
    values = case.model_dump(mode="python", exclude={"source_checksum"})
    values.update(
        gold_set_version="gold-set.v3.1",
        corpus_snapshot_id="synthetic-gold-corpus.v3.1",
        case_version="2.1",
        model_input=model_input,
    )
    view = GoldCaseV31.model_construct(**values, source_checksum="0" * 64)
    return GoldCaseV31(
        **values,
        source_checksum=payload_sha256(
            view.model_dump(mode="json", exclude={"source_checksum"})
        ),
    )


def build_bundle():
    originals, _, previous_manifest = build_gold_v3()
    cases = tuple(revised_case(case) for case in originals.cases)
    truths = tuple(derive_truth(case) for case in cases)
    cases_file = CasesFileV31(
        cases=cases,
        payload_sha256=payload_sha256(
            {"cases": [c.model_dump(mode="json") for c in cases]}
        ),
    )
    truth_file = TruthFileV11(
        truths=truths,
        payload_sha256=payload_sha256(
            {"truths": [t.model_dump(mode="json") for t in truths]}
        ),
    )
    values = previous_manifest.model_dump(mode="python", exclude={"manifest_sha256"})
    values.update(
        gold_set_version="gold-set.v3.1",
        corpus_snapshot_id="synthetic-gold-corpus.v3.1",
        generator_version="procurement-gold-generator.v3.1",
        cases_payload_sha256=cases_file.payload_sha256,
        truth_payload_sha256=truth_file.payload_sha256,
    )
    view = ManifestV31.model_construct(**values, manifest_sha256="0" * 64)
    manifest = ManifestV31(
        **values,
        manifest_sha256=payload_sha256(
            view.model_dump(mode="json", exclude={"manifest_sha256"})
        ),
    )
    validate_gold_v3(cases_file, truth_file, manifest, truth_oracle=derive_truth)
    return cases_file, truth_file, manifest


def serialized():
    return {
        name: (
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        for name, value in zip(
            ("cases.json", "truth.json", "manifest.json"), build_bundle()
        )
    }


def load_bundle(directory=DIRECTORY):
    cases = CasesFileV31.model_validate_json((directory / "cases.json").read_bytes())
    truth = TruthFileV11.model_validate_json((directory / "truth.json").read_bytes())
    manifest = ManifestV31.model_validate_json(
        (directory / "manifest.json").read_bytes()
    )
    validate_gold_v3(cases, truth, manifest, truth_oracle=derive_truth)
    return cases, truth, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for name, body in serialized().items():
        path = DIRECTORY / name
        if args.write:
            if path.exists() and path.read_bytes() != body:
                raise ValueError("existing corpus revision cannot be overwritten")
            _atomic_write(path, body)
        elif not path.exists() or path.read_bytes() != body:
            raise ValueError("corpus revision differs from generator")
    print(
        "gold-set.v3.1: DEV=120 validation=60 formal_blind=0 network_calls=0; quotas and source sufficiency verified"
    )


if __name__ == "__main__":
    main()
