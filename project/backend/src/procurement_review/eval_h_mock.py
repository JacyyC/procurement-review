"""Material-only disposable wire. No Label, Gold or Oracle is imported or opened."""

import json
import re

import httpx


def answer(task):
    source = task["sources"][0]
    text = source["content"]
    values, witnesses = {}, []

    def line(needle):
        return next((item for item in text.splitlines() if needle in item), None)

    def grab(pattern):
        match = re.search(pattern, text)
        if match is None:
            raise ValueError("h_mock_fixture_grammar")
        return match[1]

    def add(duty, field, value, clause):
        values.setdefault(duty, {})[field] = value
        start = text.index(clause)
        witnesses.append(
            dict(
                obligation_id=duty,
                field=field,
                value=value,
                source_ref=source["ref"],
                start=start,
                end=start + len(clause),
            )
        )

    if task["worker"] == "policy":
        clause = line("本办法")
        for field, value in (("applies", True), ("basis", clause), ("exempt", [])):
            add("O-01", field, value, clause)
        rule, limits = line("报价规则"), line("交付不超过")
        numbers = {
            name: int(grab(pattern))
            for name, pattern in (
                ("quotes", r"至少\s*(\d+)\s*家"),
                ("delivery", r"交付不超过(\d+)日"),
                ("acceptance", r"验收不超过(\d+)日"),
                ("warranty", r"质保至少(\d+)日"),
            )
        }
        for field, value in numbers.items():
            add("O-02", field, value, rule if field == "quotes" else limits)
        for field, value, clause in (
            (
                "special",
                f"delivery<={numbers['delivery']};acceptance<={numbers['acceptance']};warranty>={numbers['warranty']}",
                limits,
            ),
            ("supplier", "all_qualified", line("供应商要求")),
            ("money", "equal_invoice_capped_tx", line("订单与发票")),
            ("dates", "booked_ge_ordered", line("记账日期")),
        ):
            add("O-02", field, value, clause)
        add("O-04", "minimum_quotes", numbers["quotes"], rule)
        add("O-04", "applies_to", ["purchase", "service"], rule)
        exclusion, exception = line("租赁类交易不适用"), line("登记的例外")
        add(
            "O-04",
            "excluded_categories",
            ["lease"] if exclusion else [],
            exclusion or rule,
        )
        add(
            "O-04",
            "exceptions",
            ["single_source"] if exception else [],
            exception or rule,
        )
        if exclusion:
            add("O-04", "exclusion_basis", exclusion, exclusion)
        if exception:
            add(
                "O-04",
                "exception_requires",
                ["written_justification", "approval_reference"],
                exception,
            )
            add("O-04", "exception_basis", exception, exception)
    else:
        for duty, field, pattern, needle in (
            ("O-05", "buyer", r"买方BUYER:\s*(\S+?)[；。]", "买方BUYER"),
            ("O-05", "seller", r"卖方SELLER:\s*(\S+?)[；。]", "买方BUYER"),
            ("O-05", "subject", r"标的SUBJECT:\s*([^。]+)", "买方BUYER"),
            ("O-05", "order", r"订单ORDER:\s*(\S+?)[；。]", "订单ORDER"),
            ("O-05", "contract", r"合同CONTRACT:\s*(\S+?)[；。]", "订单ORDER"),
            ("O-06", "amount", r"价款记录:\s*订单\s*([\d.]+)", "价款记录"),
            ("O-06", "currency", r"价款记录:\s*订单\s*[\d.]+\s*(CNY|USD)", "价款记录"),
            ("O-08", "standard", r"验收标准STANDARD:\s*(.+?)。", "验收标准"),
            (
                "O-08",
                "responsibility",
                r"质保责任WARRANTY_RESPONSIBILITY:\s*(.+?)。",
                "质保责任",
            ),
            ("O-09", "damages", r"赔付DAMAGES:\s*(.+?)。", "赔付DAMAGES"),
            ("O-09", "cap", r"责任上限CAP:\s*([\d.]+\s+\w+)", "责任上限"),
            ("O-09", "other", r"(本次只审[^。]+)", "本次只审"),
        ):
            add(duty, field, grab(pattern), line(needle))
        for field, marker in (
            ("delivery", "DELIVERY"),
            ("acceptance", "ACCEPTANCE"),
            ("warranty", "WARRANTY"),
        ):
            add(
                "O-08",
                field,
                int(grab(marker + r"\s*(\d+)\s*日")),
                line("交付DELIVERY"),
            )
        add("O-09", "written", True, line("书面签署"))
        category = line("类交易（本次合同")
        if category:
            add(
                "O-04",
                "category",
                "lease" if "租赁类" in category else "purchase",
                category,
            )
            add("O-04", "category_basis", category, category)
        for field, needle in (
            ("single_source", "单一来源采购"),
            ("written_justification", "书面说明:"),
        ):
            clause = line(needle)
            if clause:
                add("O-04", field, True, clause)
        clause = line("审批授权:")
        if clause:
            add("O-04", "approval_reference", grab(r"审批授权:\s*(\S+?)[；。]"), clause)
        clause = line("已被补充协议替代")
        if clause:
            add("O-04", "commitment_current", False, clause)
            add("O-04", "superseding_amendment", clause, clause)
    findings = []
    for check in task["checks"]:
        duty = check["obligation_id"]
        if duty not in values:
            findings.append(
                dict(
                    obligation_id=duty,
                    status="unknown",
                    missing_information=["交易类别缺失"],
                    statement="材料不足",
                    recommended_action="补充所缺证据",
                )
            )
        else:
            findings.append(
                dict(
                    obligation_id=duty,
                    status="supported",
                    normalized_value=values[duty],
                    evidence_refs=[source["ref"]],
                    statement="原文提取",
                    recommended_action="人工核对来源",
                )
            )
    return dict(
        schema_version="finding-evidence.v3", findings=findings, witnesses=witnesses
    )


def mock_transport(mode="material", requests=None):
    counts = {}

    def respond(request):
        payload = json.loads(request.content)
        if requests is not None:
            requests.append(payload)
        if mode == "http500":
            return httpx.Response(500)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        tasks = user.get("tasks") or {user["worker"]: user}
        key = tuple((role, task["sources"][0]["ref"]) for role, task in tasks.items())
        counts[key] = counts.get(key, 0) + 1
        body = {role: answer(task) for role, task in tasks.items()}
        if "tasks" not in user:
            body = body[user["worker"]]
        content = (
            "{"
            if mode == "correction" and counts[key] == 1
            else json.dumps(body, ensure_ascii=False)
        )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-1-flash",
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        )

    return httpx.MockTransport(respond)
