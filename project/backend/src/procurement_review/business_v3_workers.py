"""Opt-in isolated worker orchestration; model calls only for two document roles."""

from .business_v3_aggregate import aggregate_business_v3
from .business_v3_documents import (
    document_request_v3,
    decode_document_values,
    DocumentValuesV3,
)
from .business_v3_procurement import review_procurement_v3


async def review_business_v3(
    context,
    *,
    case_id,
    question,
    policy_sources,
    contract_sources,
    procurement_result,
    authorized_scope_id,
    model_branch,
):
    documents = []
    for worker, sources in (("policy", policy_sources), ("contract", contract_sources)):
        request = document_request_v3(
            context, case_id=case_id, worker=worker, question=question, sources=sources
        )
        try:
            validated = await model_branch(request)
        except (TimeoutError, ConnectionError):
            documents.append(
                DocumentValuesV3(worker, {}, (), ("document_transport_failed",))
            )
        except Exception:
            # The external port may fail with arbitrary exception text. Preserve a
            # fixed failure only; cancellation (BaseException) still propagates.
            documents.append(
                DocumentValuesV3(worker, {}, (), ("document_model_failed",))
            )
        else:
            documents.append(decode_document_values(request, validated))
    procurement = review_procurement_v3(
        context, procurement_result, authorized_scope_id=authorized_scope_id
    )
    return aggregate_business_v3(context, documents[0], documents[1], procurement)
