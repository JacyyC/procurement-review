"""Clause-level synthetic documents for the Gold corpus v2.

v1 gave every case a one-line policy and contract document, so the corpus was
distinguishable only by its four-digit case number and the recommendation, conflict
and retrieval metrics had no resolution (V1-02 Evidence §42). v2 renders each case's
*already registered* fields as clause text: the document vocabulary is neutral
business attributes, and no document states a decision, a rule ID or an expectation.

Nothing here may consume the generator's shared ``random.Random`` stream; the case
signature is derived from its own digest so the business facts stay byte-identical
to v1 and the Gold Labels keep their meaning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Final

from .contracts import BusinessFact

GOLD_DOCUMENT_PROFILE_V2: Final = "gold-documents.v2"
GOLD_DOCUMENT_UNREGISTERED: Final = "未登记"

CATEGORIES: Final[tuple[str, ...]] = (
    "冷链干线保温箱",
    "矿用带式输送机",
    "阻燃工装面料",
    "光伏边框焊带",
    "色谱级乙腈试剂",
    "数控磨床主轴",
    "应急发电车租赁",
    "生鲜气调包装机",
    "盾构管片模具",
    "实验室通风柜",
    "无人机植保服务",
    "服务器托管机柜",
    "防爆对讲终端",
    "液压升降平台",
    "医用敷料耗材",
    "市政路灯控制器",
    "餐饮冷链蒸排车",
    "催化裂化助剂",
    "轨道扣件螺栓",
    "海工焊缝焊丝",
    "半导体晶圆载具",
    "风电齿轮箱油品",
    "垃圾渗滤液膜组",
    "油田注水泵壳",
    "陶瓷窑炉辊棒",
    "纺织染整助剂",
    "船舶防腐涂料",
    "铁路信号电缆",
    "森林消防水泵",
    "粮库环流熏蒸机",
    "牧场挤奶成套设备",
    "医院消毒供应台",
)
DEPARTMENTS: Final[tuple[str, ...]] = (
    "冷链仓储处",
    "装备维保科",
    "化验计量中心",
    "动力装备部",
    "行政后勤处",
    "审计监察室",
    "信息化运维组",
    "工程管理部",
    "质量检验站",
    "招标代理组",
    "供应链金融部",
    "生产计划处",
    "安全环保科",
    "燃料管理部",
    "机电仪控中心",
    "包装印刷车间",
    "港口装卸作业队",
    "矿山救护中队",
    "园林绿化所",
    "食堂事务班",
    "图书档案室",
    "车辆调度台",
    "印染前处理工段",
    "陶泥原料工段",
)
REGIONS: Final[tuple[str, ...]] = (
    "皖北",
    "赣南",
    "豫西",
    "湘西",
    "黔东南",
    "川东",
    "陕南",
    "冀东",
    "胶东",
    "浙西南",
    "粤西",
    "桂北",
    "滇东北",
    "蒙东",
    "陇南",
    "苏北",
    "闽南",
    "鄂西",
    "黔西南",
    "宁南",
)
CLAUSES: Final[tuple[str, ...]] = (
    "预付款保函",
    "到货验收",
    "质保金留存",
    "价格调整公式",
    "违约赔付上限",
    "分批交付节点",
    "变更签证流程",
    "隐蔽工程记录",
    "备件清单完整性",
    "售后响应时限",
    "履约保证金",
    "廉洁合作条款",
    "数据交接与销毁",
    "知识产权归属",
    "分包限制",
    "税率与发票类型",
    "计量器具检定",
    "冷链温控记录",
    "危化品储运资质",
    "培训与移交文档",
)


@dataclass(frozen=True, slots=True)
class GoldCaseSignature:
    """The neutral business attributes that make one case's corpus its own."""

    category: str
    department: str
    region: str
    clause: str

    @property
    def terms(self) -> tuple[str, str, str, str]:
        return (self.category, self.department, self.region, self.clause)


def _pick(values: tuple[str, ...], number: int, salt: str) -> str:
    digest = hashlib.sha256(f"{GOLD_DOCUMENT_PROFILE_V2}:{salt}:{number}".encode())
    return values[int.from_bytes(digest.digest()[:8], "big") % len(values)]


def case_signature(number: int) -> GoldCaseSignature:
    return GoldCaseSignature(
        _pick(CATEGORIES, number, "category"),
        _pick(DEPARTMENTS, number, "department"),
        _pick(REGIONS, number, "region"),
        _pick(CLAUSES, number, "clause"),
    )


def build_question_v2(number: int) -> str:
    signature = case_signature(number)
    return (
        f"请依据演示政策审查合成采购 Case {number:04d}："
        f"{signature.department}提交的{signature.region}片区"
        f"{signature.category}采购与合同变更，重点核对{signature.clause}条款，"
        "只返回有证据支持的结构化发现；不得执行工具写入，不得臆造缺失事实。"
    )


def _facts_by_tool(facts: tuple[BusinessFact, ...]) -> dict[str, dict[str, object]]:
    return {fact.tool_name: fact.result for fact in facts}


def _field(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    return GOLD_DOCUMENT_UNREGISTERED if value is None else str(value)


def _yuan(value: object) -> str:
    if value is None:
        return GOLD_DOCUMENT_UNREGISTERED
    return f"{int(value) / 100:,.2f} 元"


def build_documents_v2(
    number: int,
    facts: tuple[BusinessFact, ...],
    *,
    as_of: date,
) -> tuple[str, str]:
    """Render one case's registered fields as policy and contract clause text."""

    signature = case_signature(number)
    category = signature.category
    department = signature.department
    region = signature.region
    clause = signature.clause
    values = _facts_by_tool(facts)
    supplier = values.get("procurement.get_supplier", {})
    order = values.get("procurement.get_purchase_order", {})
    transaction = values.get("procurement.list_historical_transactions", {})
    quote_count = _field(order, "competitive_quote_count")
    contract_reference = _field(order, "contract_reference")
    counterparty = _field(transaction, "counterparty_reference")
    tax_state = (
        GOLD_DOCUMENT_UNREGISTERED
        if supplier.get("tax_identifier") is None
        else "已登记"
    )
    policy_lines = [
        f"《{category}采购与供应商专项审查规程（{clause}示范稿）》",
        (
            f"第一条 适用范围。{department}在{region}片区组织的{category}采购与合同变更，"
            f"以 {as_of} 为基准日按本规程复核；{category}以外的品类不适用本规程。"
        ),
        (
            f"第二条 供应商名册摘录。本专项登记的{category}供应商：供应商引用为"
            f"{_field(supplier, 'supplier_id')}，登记状态为"
            f"{_field(supplier, 'registry_status')}，采购准入状态为"
            f"{_field(supplier, 'procurement_status')}，尽职调查结果为"
            f"{_field(supplier, 'due_diligence_status')}，制裁筛查结果为"
            f"{_field(supplier, 'sanctions_screening')}，税号字段为{tax_state}。"
            f"{category}的合格供应商须同时具备有效税号、已完成的尽职调查与通过的筛查。"
        ),
        (
            f"第三条 比价与授权。{category}单笔采购的报价供应商数量要求不少于 3 家，"
            f"本单登记报价数量为 {quote_count} 家；采购订单须登记合同引用与交易对手"
            f"引用，本单合同引用为{contract_reference}，交易对手引用为{counterparty}。"
        ),
        (
            f"第四条 金额与币种一致性。{category}采购要求订单金额、发票金额与计价币种"
            f"逐项一致，且交易累计不得超过订单金额；本单订单金额为"
            f"{_yuan(order.get('amount_minor'))}（计价币种 {_field(order, 'currency')}），"
            f"发票金额为{_yuan(order.get('invoice_amount_minor'))}（发票币种 "
            f"{_field(order, 'invoice_currency')}），已登记交易金额为"
            f"{_yuan(transaction.get('amount_minor'))}。"
        ),
        (
            f"第五条 交付与验收。{clause}要求交付与验收留痕完整：本单下单日期为"
            f"{_field(order, 'ordered_on')}，记账日期为{_field(transaction, 'booked_on')}，"
            f"验收标准以{region}片区登记的{clause}条款为准。"
        ),
        (
            f"第六条 记录与留痕。{department}应保存{category}的询价记录、评审纪要与"
            f"{clause}复核意见；登记缺失或状态未知的事项按未知识别处理，不得凭常识补齐。"
        ),
    ]
    if "dataset_version" in order:
        policy_lines.append(
            f"第七条 版本对照。本单采购订单登记的合成数据集版本为"
            f"{_field(order, 'dataset_version')}，与本规程适用的版本须逐项比对后采用。"
        )
    policy_lines.append(
        (
            f"第{('八' if 'dataset_version' in order else '七')}条 失败处理。检索、数据读取、"
            f"版本一致性或输出合同任一环节失败时，{category}的复核结果视为未决，"
            f"须由{department}登记人工处置意见后另行处理，不得自动放行。"
        )
    )
    policy_lines.append(
        "本规程全部内容为合成演示材料，不含真实企业记录、个人信息或法律意见。"
    )
    policy = "\n".join(policy_lines)
    contract = "\n".join(
        [
            f"《{region}片区{category}采购合同条款摘录（{clause}）》",
            (
                f"第一条 合同标的。本合同标的为{category}，由{department}作为采购方在"
                f"{region}片区组织实施；合同引用为{contract_reference}，订单引用为"
                f"{_field(order, 'purchase_order_id')}，供应商引用为"
                f"{_field(supplier, 'supplier_id')}。"
            ),
            (
                f"第二条 价款与币种。合同价款为{_yuan(order.get('amount_minor'))}，"
                f"计价币种为{_field(order, 'currency')}；随附发票金额为"
                f"{_yuan(order.get('invoice_amount_minor'))}，发票币种为"
                f"{_field(order, 'invoice_currency')}；{clause}要求价款、发票与币种逐项"
                f"对应。"
            ),
            (
                f"第三条 支付与保函。{clause}约定按交付节点支付；交易对手引用为"
                f"{counterparty}，已登记交易金额为"
                f"{_yuan(transaction.get('amount_minor'))}，记账日期为"
                f"{_field(transaction, 'booked_on')}，下单日期为"
                f"{_field(order, 'ordered_on')}。"
            ),
            (
                f"第四条 验收与质保。{category}到货后由{department}按{clause}组织验收，"
                f"验收记录与质保责任须可追溯至本合同引用。"
            ),
            (
                f"第五条 变更与违约。合同变更须以书面补充协议形式登记，涉及{category}数量、"
                f"价款或{clause}的变更须重新提交复核；{region}片区的争议处理以本合同"
                f"登记条款为准。"
            ),
            (
                f"第六条 合规与资料。双方应遵守{clause}项下的廉洁、数据安全与资料销毁要求，"
                f"{category}相关的询价与评审资料保存期限按{department}制度执行。"
            ),
            (
                "第七条 其他。本合同摘录全部为合成演示内容，不含真实企业记录、个人信息"
                "或真实交易凭据，仅用于审查链路复核。"
            ),
        ]
    )
    return policy, contract
