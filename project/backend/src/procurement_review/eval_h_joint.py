"""One capable joint request with the same current extraction and grounding contracts."""

import asyncio
import json
import time
from dataclasses import replace
from datetime import UTC, datetime

import httpx

from .business_v31_review import ground_document_v31_located
from .chat_transport_v2 import AttemptResult
from .gold_v3_contracts import payload_sha256
from .model_wire_v2 import WireIssue
from .precise_branch_records import branch_record, STOP_ERROR_CODES
from .precise_run_contracts import PreciseBranchCommitV1
from .precise_run_state import commit_from_branch, restore_document
from .precise_run_transport import precise_payload
from .run_stop import MAX_RETRY_BACKOFF_SECONDS


def joint_payload(transport, inputs, facts, *, correction=None):
    roles = {
        role: precise_payload(transport.config, branch.request)
        for role, branch in inputs.items()
    }
    body = dict(roles["policy"])
    role_inputs = {
        role: json.loads(payload["messages"][1]["content"])
        for role, payload in roles.items()
    }
    definitions = {}

    def scoped(value, role):
        if isinstance(value, list):
            return [scoped(item, role) for item in value]
        if isinstance(value, dict):
            return {
                key: (
                    item.replace("#/$defs/", "#/$defs/" + role + "_")
                    if key == "$ref"
                    else scoped(item, role)
                )
                for key, item in value.items()
                if key != "$defs"
            }
        return value

    schemas = {}
    for role, payload in roles.items():
        schema = payload["response_format"]["json_schema"]["schema"]
        definitions.update(
            {
                role + "_" + name: scoped(value, role)
                for name, value in schema.get("$defs", {}).items()
            }
        )
        schemas[role] = scoped(schema, role)
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "h_joint_evidence_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["policy", "contract"],
                "properties": schemas,
                "$defs": definitions,
            },
        },
    }
    body["messages"] = [
        {
            "role": "system",
            "content": roles["policy"]["messages"][0]["content"].replace(
                "隔离文档提取器", "联合文档提取器"
            ),
        },
        {
            "role": "system",
            "content": (
                "你是单一采购审查Agent，同时执行下面policy与contract的完整任务。"
                "可以结合两域材料和只读事实联合推理；两域O-04独立保留。"
                "返回包含policy和contract两个完整答案对象的JSON，分别符合各自Schema。"
                "所有值按对应任务的授权source_ref引用，不从常识补充未提供证据。"
                "采购事实由相同确定性规则聚合；无外部写入或审批权限。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"tasks": role_inputs, "readonly_facts": facts}, ensure_ascii=False
            ),
        },
    ]
    if correction:
        body["messages"].append(
            {
                "role": "system",
                "content": "请纠正这些固定合同/引用错误：" + json.dumps(correction),
            }
        )
    return body


def receipt(result, elapsed, error=None):
    return dict(
        error_code=error or result.error_code,
        correctable=result.correctable,
        retryable=result.retryable,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        usage_unknown=result.usage_unknown,
        returned=result.received_body,
        latency_ms=elapsed,
        resolved_model=result.resolved_model_id,
        http_status=result.http_status,
        finish_reason=result.finish_reason,
    )


async def execute(
    journal,
    plan,
    case,
    seed,
    transport,
    inputs,
    procurement_read,
    admit,
    *,
    stop=None,
    deadline_seconds=None,
):
    """One joint branch inside the Run deadline and the batch stop gate.

    ``deadline_seconds`` bounds the whole pass over this branch — the same total budget the
    product arm's Run is created with — so each wait is capped by the remaining real time
    and a retry backoff spends that budget instead of resetting it.  ``stop`` is the batch
    send gate: once the batch has stopped, no further physical attempt is admitted.
    """

    begin_run = time.monotonic()

    def remaining_seconds():
        if deadline_seconds is None:
            return None
        return deadline_seconds - (time.monotonic() - begin_run)

    attempts = journal.attempts(case.case_id, seed, "single_agent")
    for attempt in attempts:
        if attempt["status"] == "started":
            raise ValueError("h_unsettled_joint_attempt_no_resend")
        if attempt.get("projection") is not None:
            return {
                role: restore_document(
                    PreciseBranchCommitV1.model_validate_json(
                        json.dumps(body), strict=True
                    ),
                    inputs[role].request,
                )
                for role, body in attempt["projection"].items()
            }, attempts
    if attempts and attempts[-1]["receipt"]["error_code"] == "interrupted_unknown":
        return None, attempts
    correction = attempts[-1]["receipt"].get("correction") if attempts else None
    grounded = None
    for _ in range(2 - len(attempts)):
        if stop is not None:
            stop.gate()
        wait_seconds = transport.config.single_attempt_seconds
        remaining = remaining_seconds()
        if remaining is not None:
            if remaining <= 0.0:
                # The branch's own budget is spent; no further physical attempt starts.
                break
            wait_seconds = min(wait_seconds, remaining)
        payload = joint_payload(
            transport,
            inputs,
            [fact.model_dump(mode="json") for fact in procurement_read.facts],
            correction=correction,
        )
        admit()
        path = journal.reserve(
            case.case_id, seed, "single_agent", "joint", payload_sha256(payload)
        )
        begin = time.monotonic()
        try:
            response = await transport.post_once(payload, wait_seconds)
            admit()
        except BaseException:
            journal.recover_unknown(case.case_id, seed, "single_agent")
            raise
        errors, documents = [], {}
        if isinstance(response, AttemptResult):
            result = response
            errors = [result.error_code]
        else:
            outer = {}
            boundary = None
            try:
                outer = response.json()
                if type(outer) is not dict:
                    raise ValueError("h_joint_envelope")
                from .model_wire_v2 import strict_json

                transport.protection.assert_safe(
                    response.text, transport.credential.get_secret_value()
                )
                content = strict_json(outer["choices"][0]["message"]["content"])
                if type(content) is not dict or set(content) != set(inputs):
                    raise ValueError("h_joint_roles")
            except WireIssue as issue:
                boundary = issue.code
                content = {}
            except (ValueError, KeyError, TypeError, IndexError):
                content = {}
                if type(outer) is not dict:
                    outer = {}
            results = []
            choices = outer.get("choices")
            choice = (
                choices[0]
                if type(choices) is list and choices and type(choices[0]) is dict
                else {}
            )
            message = (
                choice.get("message") if type(choice.get("message")) is dict else {}
            )
            for role, branch in inputs.items():
                role_outer = {
                    **outer,
                    "choices": [
                        {
                            **choice,
                            "message": {
                                **message,
                                "content": json.dumps(
                                    content.get(role), ensure_ascii=False
                                ),
                            },
                        }
                    ],
                }
                # The per-role shim keeps the supplier's bounded retry signal: without it
                # a rate-limited joint attempt would retry with no wait at all, spending
                # the budget instead of the server's own Retry-After.
                retry_headers = (
                    {"retry-after": response.headers["retry-after"]}
                    if "retry-after" in response.headers
                    else None
                )
                role_response = httpx.Response(
                    response.status_code,
                    json=role_outer,
                    request=response.request,
                    headers=retry_headers,
                )
                result = transport.decode(
                    role_response,
                    branch.request,
                    transport.credential.get_secret_value(),
                )
                results.append(result)
                if result.error_code:
                    errors.append(result.error_code)
                else:
                    document, locations = ground_document_v31_located(
                        branch.request, *result.evidence, branch.catalog
                    )
                    documents[role] = document
                    errors.extend(document.support_failures)
            result = results[0]
            if boundary:
                errors = [boundary]
            result = replace(
                result,
                correctable=not boundary and any(item.correctable for item in results),
                retryable=not boundary and any(item.retryable for item in results),
            )
        elapsed = int((time.monotonic() - begin) * 1000)
        success = not errors and set(documents) == set(inputs)
        projection = (
            {
                role: commit_from_branch(
                    worker=role,
                    branch=branch_record(
                        worker=role,
                        document=document,
                        request_sha=payload_sha256(payload),
                        answer_sha=payload_sha256(content.get(role)),
                        attempts=(),
                        hits=inputs[role].hits,
                    ),
                    document=document,
                    security=(),
                    blocked=False,
                ).model_dump(mode="json")
                for role, document in documents.items()
            }
            if success
            else None
        )
        projected_receipt = receipt(result, elapsed, errors[0] if errors else None)
        projected_receipt["correction"] = sorted(set(errors)) if errors else None
        journal.settle(path, projected_receipt, projection=projection)
        attempts = journal.attempts(case.case_id, seed, "single_agent")
        if stop is not None and errors:
            stop.note_attempt(errors[0], datetime.now(UTC))
        if success:
            grounded = documents
            break
        fatal = set(errors) & STOP_ERROR_CODES
        if fatal:
            # A batch-level refusal (identity, credential, drift) is already reported to
            # the shared stop decision through the attempt receipt; a controlled refusal
            # is this observation's own measured failure.  Either way the branch stops
            # here and the batch winds down at its next gate instead of cancelling the
            # other cases' admitted requests.
            break
        citation_failure = any(str(error).startswith("citation_") for error in errors)
        retryable = bool(result.retryable)
        if not (result.correctable or citation_failure or retryable):
            break
        if result.correctable or citation_failure:
            correction = sorted(set(errors))
            continue
        # A known recoverable transport failure re-sends the same payload without a
        # correction, inside the same two-attempt budget and the remaining real deadline.
        retry_after = float(getattr(result, "retry_after", 0.0) or 0.0)
        if retry_after > MAX_RETRY_BACKOFF_SECONDS:
            break
        remaining = remaining_seconds()
        if remaining is not None:
            if remaining <= 0.0:
                break
            retry_after = min(retry_after, remaining)
        if retry_after > 0.0:
            await asyncio.sleep(retry_after)
    return grounded, attempts
