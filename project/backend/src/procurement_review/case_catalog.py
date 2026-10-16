"""The frozen V1-01 case catalog and the explicit V1-03 business fixture mapping.

The nine predeclared synthetic scenarios stay exactly as accepted in V1-01; V1-03 only
adds the tenant/Case ownership they are registered under.
"""

from __future__ import annotations

from .api_contracts import CaseOption, ReviewScenario
from .contracts import CaseType

CASE_CATALOG: tuple[CaseOption, ...] = (
    CaseOption(
        case_id="CASE-LOW-001",
        label="低风险完整证据",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.LOW,
        description="三类只读调查完成，生成待人工确认的通过建议。",
    ),
    CaseOption(
        case_id="CASE-HIGH-001",
        label="高风险政策例外",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.HIGH,
        description="政策调查返回高风险例外，聚合器建议拒绝。",
    ),
    CaseOption(
        case_id="CASE-CONFLICT-001",
        label="证据结论冲突",
        case_type=CaseType.CONTRACT_CHANGE,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.CONFLICT,
        description="政策与合同分支产生相反断言，必须人工复核。",
    ),
    CaseOption(
        case_id="CASE-UNKNOWN-001",
        label="关键证据缺失",
        case_type=CaseType.CONTRACT_CHANGE,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.UNKNOWN,
        description="检索无有效命中，系统失败关闭为信息不足。",
    ),
    CaseOption(
        case_id="CASE-PARTIAL-FAILURE-001",
        label="单 Worker 失败",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.PARTIAL_FAILURE,
        description="合同 Worker 的测试 Fixture 无效，其余分支仍可核验。",
    ),
    CaseOption(
        case_id="CASE-ALL-FAILURE-001",
        label="全部 Worker 失败",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.ALL_FAILURE,
        description="三个 Worker 均失败，系统不会给出普通审批建议。",
    ),
    CaseOption(
        case_id="CASE-VERSION-001",
        label="数据版本不匹配",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.VERSION_MISMATCH,
        description="检索与采购数据版本过期，所有错误保持可见。",
    ),
    CaseOption(
        case_id="CASE-CANCEL-001",
        label="处理中取消",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.CANCELLABLE,
        description="本地测试流程短暂等待，用于验证处理中取消语义。",
    ),
    CaseOption(
        case_id="CASE-RUNTIME-FAILURE-001",
        label="运行时失败",
        case_type=CaseType.PURCHASE_ORDER,
        purchase_order_id="PO-001",
        scenario=ReviewScenario.RUNTIME_FAILURE,
        description="注入本地运行故障，Run 进入 failed 且不生成回执。",
    ),
)

CASE_CATALOG_BY_ID = {item.case_id: item for item in CASE_CATALOG}

__all__ = ["CASE_CATALOG", "CASE_CATALOG_BY_ID"]
