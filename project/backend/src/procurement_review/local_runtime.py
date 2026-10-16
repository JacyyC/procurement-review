"""Local-only runtime assembly for the Review API.

The runtime intentionally uses the package's declared deterministic fixtures and
an in-memory procurement reader behind the existing official MCP server.  It is
never a production persistence or model integration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp import Client

from .api_contracts import (
    DEFAULT_PURCHASE_ORDER_ID,
    ReviewCreateRequest,
    ReviewScenario,
)
from .ingestion import (
    DEFAULT_CONFIG,
    IngestedChunk,
    LocatorKind,
    SourceLocator,
    make_chunk_id,
    make_document_id,
    make_source_version_id,
)
from .mcp_server import create_procurement_mcp_server
from .model_adapters import DeterministicFakeModelAdapter
from .procurement_mcp_client import OfficialProcurementMCPClient
from .procurement_store import (
    ProcurementDatasetIdentity,
    ProcurementReader,
    ProcurementRecord,
    read_fixture,
)
from .retrieval import (
    DeterministicFakeVectorAdapter,
    DeterministicHybridRetriever,
    DeterministicLexicalIndex,
    RetrievalPort,
)
from .checkpointing import (
    BRANCH_TIMEOUT_SECONDS,
    BranchTelemetry,
    CheckpointedPorts,
    ReadAdmission,
)
from .fault_hooks import FaultController
from .model_invocation import OfflineModelInvocation, RunModelInvocation
from .result_admission import OFFLINE_RESULT_CONSUMPTION, RunResultAdmission
from .review_graph import ReviewExecution, ReviewInvocation, ReviewWorkflow
from .review_repository import ReviewRepository
from .review_store import CaseRegistration, ReviewStore
from .rules import RULE_SET, RULE_SET_V2, RULE_SETS, RuleSet
from .review_service import IncompatibleCheckpointFailure
from .scoped_runtime import ScopeIntegrityError, ScopedReviewRuntime
from .security_contracts import RuntimeScope
from .worker_contracts import (
    ContractWorkerInput,
    PolicyWorkerInput,
    ProcurementWorkerInput,
)
from .workers import ContractWorker, PolicyWorker, ProcurementAnomalyWorker

RETRIEVAL_FIXTURE_PATH = (
    Path(__file__).parent / "resources" / "retrieval" / "fixture.v1.json"
)
LOCAL_SOURCE_VERSION = "Generated Retrieval Fixture, Version 1"


def _checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _fixture_chunks() -> tuple[IngestedChunk, ...]:
    fixture = json.loads(RETRIEVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    chunks: list[IngestedChunk] = []
    for document in fixture["documents"]:
        source_id = str(document["source_id"])
        content = str(document["content"])
        source_bytes = content.encode("utf-8")
        source_checksum = _checksum(source_bytes)
        source_version_id = make_source_version_id(source_id, LOCAL_SOURCE_VERSION)
        document_id = make_document_id(
            source_id=source_id,
            source_version=LOCAL_SOURCE_VERSION,
            source_checksum=source_checksum,
            media_type="text/plain",
            ingestion_version=DEFAULT_CONFIG.ingestion_version,
        )
        locator = SourceLocator(
            kind=LocatorKind.TEXT_LINES,
            line_start=1,
            line_end=max(1, content.count("\n") + 1),
        )
        chunk_id = make_chunk_id(
            document_id=document_id,
            ordinal=1,
            locator=locator,
            content_checksum=source_checksum,
            chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
            chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
        )
        chunks.append(
            IngestedChunk(
                source_id=source_id,
                source_version=LOCAL_SOURCE_VERSION,
                source_version_id=source_version_id,
                source_checksum=source_checksum,
                document_id=document_id,
                chunk_id=chunk_id,
                ordinal=1,
                media_type="text/plain",
                locator=locator,
                content=content,
                content_checksum=source_checksum,
                ingestion_version=DEFAULT_CONFIG.ingestion_version,
                chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
                chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
                index_version=DEFAULT_CONFIG.index_version,
            )
        )
    return tuple(chunks)


def build_local_retriever() -> RetrievalPort:
    """Build the deterministic retrieval fixture without network or disk writes."""

    fixture = json.loads(RETRIEVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    chunks = _fixture_chunks()
    ids_by_source = {chunk.source_id: chunk.chunk_id for chunk in chunks}
    vector_scores = {
        query["query"]: {
            ids_by_source[source_id]: score
            for source_id, score in query["fake_vector_scores"].items()
        }
        for query in fixture["queries"]
    }
    return DeterministicHybridRetriever(
        lexical_index=DeterministicLexicalIndex(
            chunks,
            index_version=DEFAULT_CONFIG.index_version,
        ),
        vector_port=DeterministicFakeVectorAdapter(
            index_version=DEFAULT_CONFIG.index_version,
            scores_by_query=vector_scores,
        ),
    )


class LocalSyntheticProcurementReader:
    """Read-only in-memory view of the pinned synthetic fixture."""

    test_only = True

    def __init__(self) -> None:
        fixture = read_fixture()
        payload = fixture.payload
        self._identity = ProcurementDatasetIdentity(
            dataset_version=payload.dataset_version,
            fixture_schema_version=payload.fixture_schema_version,
            envelope_schema_version=fixture.envelope_schema_version,
            generator_version=payload.generator_version,
            seed=payload.seed,
            as_of=payload.as_of,
            digest_algorithm=fixture.digest_algorithm,
            payload_canonicalization=fixture.payload_canonicalization,
            payload_sha256=fixture.payload_sha256,
        )
        self._suppliers = {
            item.supplier_id: item.model_dump(mode="json") for item in payload.suppliers
        }
        self._orders = {
            item.purchase_order_id: item.model_dump(mode="json")
            for item in payload.purchase_orders
        }
        self._transactions = tuple(
            item.model_dump(mode="json") for item in payload.historical_transactions
        )

    @property
    def identity(self) -> ProcurementDatasetIdentity:
        return self._identity

    def _check_version(self, dataset_version: str) -> None:
        if dataset_version != self.identity.dataset_version:
            from .procurement_store import DatasetVersionMismatchError

            raise DatasetVersionMismatchError("synthetic dataset version mismatch")

    def get_supplier(
        self, supplier_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        self._check_version(dataset_version)
        payload = self._suppliers.get(supplier_id)
        if payload is None:
            return None
        return ProcurementRecord(
            record_id=supplier_id,
            dataset_version=dataset_version,
            as_of=self.identity.as_of,
            payload=dict(payload),
        )

    def get_purchase_order(
        self, purchase_order_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        self._check_version(dataset_version)
        payload = self._orders.get(purchase_order_id)
        if payload is None:
            return None
        return ProcurementRecord(
            record_id=purchase_order_id,
            dataset_version=dataset_version,
            as_of=self.identity.as_of,
            payload=dict(payload),
        )

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> tuple[ProcurementRecord, ...]:
        self._check_version(dataset_version)
        if (purchase_order_id is None) == (supplier_id is None):
            from .procurement_store import InvalidProcurementQueryError

            raise InvalidProcurementQueryError("select exactly one history scope")
        rows = (
            item
            for item in self._transactions
            if (
                purchase_order_id is not None
                and item["purchase_order_id"] == purchase_order_id
            )
            or (supplier_id is not None and item["supplier_id"] == supplier_id)
        )
        return tuple(
            ProcurementRecord(
                record_id=item["transaction_id"],
                dataset_version=dataset_version,
                as_of=self.identity.as_of,
                payload=dict(item),
            )
            for item in rows
        )


@dataclass(frozen=True, slots=True)
class ScenarioInputs:
    policy_query: str
    contract_query: str
    policy_fixture_id: str
    contract_fixture_id: str
    procurement_fixture_id: str
    index_version: str
    dataset_version: str


def scenario_inputs(request: ReviewCreateRequest) -> ScenarioInputs:
    """Map a predeclared UI scenario to existing worker fixture IDs."""

    scenario = request.scenario
    values = {
        "policy_query": request.policy_query,
        "contract_query": request.contract_query,
        "policy_fixture_id": request.policy_model_fixture_id or "policy-low",
        "contract_fixture_id": request.contract_model_fixture_id or "contract-low",
        "procurement_fixture_id": request.procurement_model_fixture_id
        or "procurement-low",
        "index_version": request.index_version,
        "dataset_version": request.dataset_version,
    }
    if scenario is ReviewScenario.HIGH:
        values["policy_fixture_id"] = request.policy_model_fixture_id or "policy-high"
    elif scenario is ReviewScenario.CONFLICT:
        values["policy_fixture_id"] = (
            request.policy_model_fixture_id or "policy-conflict-present"
        )
        values["contract_fixture_id"] = (
            request.contract_model_fixture_id or "contract-conflict-absent"
        )
    elif scenario is ReviewScenario.UNKNOWN:
        values["policy_query"] = "unrepresented phrase"
    elif scenario is ReviewScenario.PARTIAL_FAILURE:
        values["contract_fixture_id"] = (
            request.contract_model_fixture_id or "contract-fixture-not-declared"
        )
    elif scenario is ReviewScenario.ALL_FAILURE:
        values["policy_fixture_id"] = (
            request.policy_model_fixture_id or "policy-fixture-not-declared"
        )
        values["contract_fixture_id"] = (
            request.contract_model_fixture_id or "contract-fixture-not-declared"
        )
        values["procurement_fixture_id"] = (
            request.procurement_model_fixture_id or "procurement-fixture-not-declared"
        )
    elif scenario is ReviewScenario.VERSION_MISMATCH:
        values["index_version"] = "retrieval-index-v999"
        values["dataset_version"] = "synthetic-procurement.v999"
    if request.document_version != "retrieval-fixture-v1":
        values["index_version"] = "retrieval-index-v999"
    return ScenarioInputs(**values)


class LocalWorkflowRunner:
    """Legacy in-memory runner kept for offline unit assertions only.

    The product API never uses this class: it has no runtime scope and therefore no
    place in the authenticated short path.  It remains for the frozen V1-01/V1-05
    offline contracts that compare rule sets without a database.
    """

    def __init__(self, *, rule_set: RuleSet = RULE_SET) -> None:
        self._retrieval = build_local_retriever()
        self._reader: ProcurementReader = LocalSyntheticProcurementReader()
        self._model = DeterministicFakeModelAdapter()
        self._rule_set = rule_set

    async def run(
        self,
        *,
        run_id: str,
        request: ReviewCreateRequest,
    ) -> ReviewExecution:
        if request.scenario is ReviewScenario.RUNTIME_FAILURE:
            raise RuntimeError("injected local test-only runtime failure")
        if request.scenario is ReviewScenario.CANCELLABLE:
            await asyncio.sleep(0.5)
        inputs = scenario_inputs(request)
        # No Run to re-authorize and no review database: this offline runner states its
        # model-call policy by name instead of being handed a silent pass.
        invocation = OfflineModelInvocation(receiver=self._model)
        policy = PolicyWorker(
            retrieval=self._retrieval,
            model=self._model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=invocation,
        )
        contract = ContractWorker(
            retrieval=self._retrieval,
            model=self._model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=invocation,
        )
        server = create_procurement_mcp_server(
            self._reader,
            scope=unscoped_test_scope(request),
        )
        async with Client(server, raise_exceptions=True) as client:
            procurement = ProcurementAnomalyWorker(
                mcp=OfficialProcurementMCPClient(client),
                model=self._model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=invocation,
            )
            workflow = ReviewWorkflow(
                policy_worker=policy,
                contract_worker=contract,
                procurement_worker=procurement,
                rule_set=self._rule_set,
            )
            invocation = ReviewInvocation(
                run_id=run_id,
                case_id=request.case_id,
                policy_input=PolicyWorkerInput(
                    case_id=request.case_id,
                    query=inputs.policy_query,
                    index_version=inputs.index_version,
                    model_fixture_id=inputs.policy_fixture_id,
                ),
                contract_input=ContractWorkerInput(
                    case_id=request.case_id,
                    query=inputs.contract_query,
                    index_version=inputs.index_version,
                    model_fixture_id=inputs.contract_fixture_id,
                ),
                procurement_input=ProcurementWorkerInput(
                    case_id=request.case_id,
                    purchase_order_id=request.purchase_order_id
                    or DEFAULT_PURCHASE_ORDER_ID,
                    dataset_version=inputs.dataset_version,
                    model_fixture_id=inputs.procurement_fixture_id,
                ),
            )
            return await workflow.run(invocation)


def unscoped_test_scope(request: ReviewCreateRequest) -> RuntimeScope:
    """Build an explicit fixture scope for the offline legacy runner.

    This is *not* an implicit unrestricted default: the caller must state the case,
    the procurement record set and the dataset version, and the resulting scope is a
    normal :class:`RuntimeScope` that the MCP server always enforces.
    """

    fixture = read_fixture()
    payload = fixture.payload
    order_id = request.purchase_order_id or DEFAULT_PURCHASE_ORDER_ID
    order = next(
        item for item in payload.purchase_orders if item.purchase_order_id == order_id
    )
    supplier_ids = tuple(sorted({order.supplier_id}))
    records = _fixture_chunks()
    return RuntimeScope(
        tenant_id="tenant-offline-fixture",
        case_id=request.case_id,
        case_version=f"{request.case_id.lower()}.v1",
        case_type=request.case_type.value,
        scenario=request.scenario.value,
        subject_id="subject-offline-fixture",
        credential_id="credential-offline-fixture",
        authorization_revision=1,
        generation=1,
        document_version=request.document_version,
        dataset_version=payload.dataset_version,
        index_version=DEFAULT_CONFIG.index_version,
        rules_version=RULE_SET_V2.rules_version,
        case_document_ids=(),
        shared_document_ids=tuple(sorted(item.document_id for item in records)),
        allowed_order_ids=(order_id,),
        allowed_supplier_ids=supplier_ids,
        allowed_transaction_ids=tuple(
            sorted(
                item.transaction_id
                for item in payload.historical_transactions
                if item.purchase_order_id == order_id
            )
        ),
        issued_at=datetime.now(UTC),
        deadline_at=datetime.now(UTC) + timedelta(seconds=60),
    )


class ScopedWorkflowRunner:
    """Assemble one persisted, checkpoint-aware review execution.

    No adapter is constructed before the scope exists, every Worker receives only the
    ports the scope produced, and each branch runs behind a checkpoint-aware wrapper
    that reuses an adopted result instead of reading again.
    """

    def __init__(
        self,
        *,
        store: ReviewStore,
        branch_timeout_seconds: float = BRANCH_TIMEOUT_SECONDS,
        model: object | None = None,
        evidence_verifier: Callable[[ReviewExecution], None] | None = None,
    ) -> None:
        self._store = store
        self._branch_timeout_seconds = branch_timeout_seconds
        self._model = model or DeterministicFakeModelAdapter()
        self._evidence_verifier = evidence_verifier
        self.telemetry = BranchTelemetry()

    def build_model_invocation(
        self,
        *,
        run_id: str,
        generation: int,
        owner_id: str,
        now: Callable[[], datetime],
        repository: ReviewRepository,
        telemetry: BranchTelemetry,
        fault: FaultController | None = None,
    ) -> RunModelInvocation:
        """Build the one model-call boundary this execution's Workers must use.

        This is the production assembly of the boundary: it binds the Run, the generation,
        the live-authority repository and the adapter that receives the request.  It is a
        method rather than an inline expression so that an observation can wrap the very
        boundary the Workers call instead of standing in for it — and so that the one
        production path stays the only place a boundary is built for a persisted Run.
        """

        return RunModelInvocation(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            receiver=self._model,
            telemetry=telemetry,
            fault=fault,
        )

    async def run(
        self,
        *,
        run_id: str,
        scope: RuntimeScope,
        registration: CaseRegistration,
        generation: int,
        owner_id: str,
        now: Callable[[], datetime],
        repository: ReviewRepository,
        deadline_at: datetime,
        fault: FaultController | None = None,
        heartbeat: Callable[[], bool] | None = None,
    ) -> ReviewExecution:
        del deadline_at, heartbeat
        scenario = ReviewScenario(registration.scenario)
        if scenario is ReviewScenario.RUNTIME_FAILURE:
            # The registered Case asks for the local fault injection; no client field,
            # query parameter or header can request it.
            raise RuntimeError("injected local test-only runtime failure")
        if scenario is ReviewScenario.CANCELLABLE and not _is_resume(
            repository, run_id
        ):
            await asyncio.sleep(0.5)
        admission = ReadAdmission(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            telemetry=self.telemetry,
            fault=fault,
        )
        # One consumption admission per execution, bound to this Run and this generation:
        # a read that has already returned and a model answer that has already come back are
        # both re-admitted against the live authority before the Worker may use them.
        consumption = RunResultAdmission(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            telemetry=self.telemetry,
            fault=fault,
        )
        # One model-call boundary per execution, bound to the same Run and generation: the
        # request factory runs only after the live authority has been re-read, and the
        # adapter receives the request inside that same short transaction.
        invocation = self.build_model_invocation(
            run_id=run_id,
            generation=generation,
            repository=repository,
            telemetry=self.telemetry,
            fault=fault,
            now=now,
            owner_id=owner_id,
        )
        runtime = ScopedReviewRuntime.build(
            store=self._store,
            scope=scope,
            admission=admission,
        )
        inputs = scoped_scenario_inputs(registration)
        rule_set = RULE_SETS[scope.rules_version]
        ports = CheckpointedPorts(
            repository=repository,
            run_id=run_id,
            scope=scope,
            generation=generation,
            now=now,
            owner_id=owner_id,
            fault=fault,
            telemetry=self.telemetry,
        )
        policy = ports.wrap(
            "policy",
            PolicyWorker(
                retrieval=runtime.retrieval,
                model=self._model,
                consumption=consumption,
                invocation=invocation,
            ),
        )
        contract = ports.wrap(
            "contract",
            ContractWorker(
                retrieval=runtime.retrieval,
                model=self._model,
                consumption=consumption,
                invocation=invocation,
            ),
        )
        server = create_procurement_mcp_server(runtime.procurement, scope=scope)
        async with Client(server, raise_exceptions=True) as client:
            procurement = ports.wrap(
                "procurement_data",
                ProcurementAnomalyWorker(
                    mcp=OfficialProcurementMCPClient(client),
                    model=self._model,
                    consumption=consumption,
                    invocation=invocation,
                ),
            )
            workflow = ReviewWorkflow(
                policy_worker=policy,
                contract_worker=contract,
                procurement_worker=procurement,
                branch_timeout_seconds=self._branch_timeout_seconds,
                rule_set=rule_set,
            )
            invocation = ReviewInvocation(
                run_id=run_id,
                case_id=registration.case_id,
                policy_input=PolicyWorkerInput(
                    case_id=registration.case_id,
                    query=inputs.policy_query,
                    index_version=inputs.index_version,
                    model_fixture_id=inputs.policy_fixture_id,
                ),
                contract_input=ContractWorkerInput(
                    case_id=registration.case_id,
                    query=inputs.contract_query,
                    index_version=inputs.index_version,
                    model_fixture_id=inputs.contract_fixture_id,
                ),
                procurement_input=ProcurementWorkerInput(
                    case_id=registration.case_id,
                    purchase_order_id=registration.purchase_order_id,
                    dataset_version=inputs.dataset_version,
                    model_fixture_id=inputs.procurement_fixture_id,
                ),
            )
            execution = await workflow.run(invocation)
        try:
            runtime.verify_execution(execution)
        except ScopeIntegrityError as error:
            raise IncompatibleCheckpointFailure(str(error)) from error
        if self._evidence_verifier is not None:
            self._evidence_verifier(execution)
        if fault is not None and all(
            repository.adopted_checkpoint(run_id, worker) is not None
            for worker in ("policy", "contract", "procurement_data")
        ):
            fault.barrier("after_all_branches_committed")
        return execution


def _is_resume(repository: ReviewRepository, run_id: str) -> bool:
    run = repository.load_run(run_id)
    return run is not None and run.header.resume_count > 0


def scoped_scenario_inputs(registration: CaseRegistration) -> ScenarioInputs:
    """Map a registered Case to the frozen worker fixture IDs of its scenario."""

    scenario = ReviewScenario(registration.scenario)
    values = {
        "policy_query": "competitive bidding threshold",
        "contract_query": "payment terms termination rights",
        "policy_fixture_id": "policy-low",
        "contract_fixture_id": "contract-low",
        "procurement_fixture_id": "procurement-low",
        "index_version": registration.index_version,
        "dataset_version": registration.dataset_version,
    }
    if scenario is ReviewScenario.HIGH:
        values["policy_fixture_id"] = "policy-high"
    elif scenario is ReviewScenario.CONFLICT:
        values["policy_fixture_id"] = "policy-conflict-present"
        values["contract_fixture_id"] = "contract-conflict-absent"
    elif scenario is ReviewScenario.UNKNOWN:
        values["policy_query"] = "unrepresented phrase"
    elif scenario is ReviewScenario.PARTIAL_FAILURE:
        values["contract_fixture_id"] = "contract-fixture-not-declared"
    elif scenario is ReviewScenario.ALL_FAILURE:
        values["policy_fixture_id"] = "policy-fixture-not-declared"
        values["contract_fixture_id"] = "contract-fixture-not-declared"
        values["procurement_fixture_id"] = "procurement-fixture-not-declared"
    elif scenario is ReviewScenario.VERSION_MISMATCH:
        values["index_version"] = "retrieval-index-v999"
        values["dataset_version"] = "synthetic-procurement.v999"
    return ScenarioInputs(**values)
