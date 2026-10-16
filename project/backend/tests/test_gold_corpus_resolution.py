"""Retrieval resolution gate: v1 is anchored, v2 must be measurable.

The real V1-02C2 batch may only start once the lexical retrieval probe on the
clause-level corpus shows the metrics gained resolution, so both numbers are pinned
here: v1 stays at the degenerate ``20/440`` and ``5/440`` recorded in Evidence
§40/§42, and v2 must clear 90% on both query variants. Zero model calls.
"""

from __future__ import annotations

from procurement_review.eval_metrics import RECALL_K, retrieval_report
from procurement_review.gold_corpus import build_gold_lexical_corpus
from procurement_review.gold_set import (
    GoldCorpusVersion,
    GoldSplit,
    load_gold_cases,
    load_gold_labels_for_split,
)


def _recall_at_5(version: GoldCorpusVersion, variant: str) -> tuple[int, int]:
    cases = [
        case
        for case in load_gold_cases(corpus_version=version).cases
        if case.split is GoldSplit.DEV
    ]
    corpus = build_gold_lexical_corpus(cases, split=GoldSplit.DEV)
    labels = {
        label.case_id: label
        for label in load_gold_labels_for_split(GoldSplit.DEV, corpus_version=version)
    }
    report = retrieval_report([("dev", corpus)], cases, labels, k=RECALL_K)
    statistic = report["dev"]["variants"][variant]["recall_at_5"]
    return statistic.numerator, statistic.denominator


def test_v1_dev_corpus_still_measures_the_degenerate_anchor_numbers() -> None:
    assert _recall_at_5(GoldCorpusVersion.V1, "raw_query") == (20, 440)
    assert _recall_at_5(GoldCorpusVersion.V1, "masked_case_number_query") == (5, 440)


def test_v2_dev_corpus_lets_a_lexical_retriever_find_both_gold_documents() -> None:
    raw_found, total = _recall_at_5(GoldCorpusVersion.V2, "raw_query")
    masked_found, masked_total = _recall_at_5(
        GoldCorpusVersion.V2, "masked_case_number_query"
    )
    assert total == masked_total == 440
    assert raw_found / total > 0.9
    # The masked variant drops every digit, so this can only be carried by the
    # clause-level subject matter, not by the case number.
    assert masked_found / masked_total > 0.9


def test_v2_duplicate_query_diagnostic_stays_clean() -> None:
    cases = [
        case
        for case in load_gold_cases(corpus_version=GoldCorpusVersion.V2).cases
        if case.split is GoldSplit.DEV
    ]
    corpus = build_gold_lexical_corpus(cases, split=GoldSplit.DEV)
    labels = {
        label.case_id: label
        for label in load_gold_labels_for_split(
            GoldSplit.DEV, corpus_version=GoldCorpusVersion.V2
        )
    }
    report = retrieval_report([("dev", corpus)], cases, labels, k=RECALL_K)
    duplicate = report["dev"]["duplicate_query_rate"]
    assert duplicate.numerator == 0
    assert duplicate.denominator == 220
