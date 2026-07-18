"""
FastAPI Server for Federated Learning Coordinator

Main entry point for the coordinator API.
"""

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from pathlib import Path

from core.round_manager import RoundManager
from persistence.json_repos import get_round_repository
from core.task_assigner import TaskAssigner
from core.update_validator import UpdateValidator
from core.aggregator import Aggregator
from core.model_store import ModelStore
from core.metrics import MetricsCollector
from core.auth import (
    AuthManager,
    ClientAlreadyRegisteredError,
    validate_operator_key,
    get_operator_api_key,
)
from core.rate_limiter import RateLimiter
from core.privacy import PrivacyProtector
from core.async_round_manager import AsyncRoundManager, AsyncRoundConfig
from core.reputation import ReputationManager
from core.incentives import IncentiveManager
from core.state_store import StateStore
from jobs import get_job_queue
from core.local_launcher import get_local_launcher
from core.geo_presence import get_geo_presence
from model_registry.base_models import BaseModelRegistry
from rounds import create_lora_round, get_lora_round, close_lora_round
from rounds.create_round import get_lora_round_manager
from aggregation import aggregate_lora_adapters, validate_adapter
from aggregation.adapter_manifest import build_adapter_manifest, register_adapter_manifest
from evaluation import evaluate_adapter
from core.versioning import next_version
from utils.logger import setup_coordinator_logger
import os
import json
import hashlib
import time
import threading

# Set up logging
logger = setup_coordinator_logger()
logger.info("Coordinator starting", extra={
    "component": "coordinator",
    "event": "coordinator_started"
})

if os.getenv("REQUIRE_OPERATOR_KEY", "").strip().lower() in ("1", "true", "yes"):
    if get_operator_api_key() is None:
        raise RuntimeError(
            "REQUIRE_OPERATOR_KEY is set but OPERATOR_API_KEY is empty. "
            "Refuse to start an open control plane."
        )


# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Coordinator",
    description="MVP Coordinator API for Federated Learning",
    version="1.0.0"
)

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Durable state + core modules
state_store = StateStore()
model_store = ModelStore()
round_manager = RoundManager(
    state_store=state_store,
    round_repo=get_round_repository(),
)
task_assigner = TaskAssigner(round_manager, model_store)
auth_manager = AuthManager(state_store=state_store)

_ha_reputation_repo = None
_ha_incentive_repo = None
_ha_geo_repo = None
_ha_rate_repo = None
try:
    from persistence.shared_state import shared_state_enabled
    from persistence.db import create_all_tables

    if shared_state_enabled():
        create_all_tables()
        from persistence.ha_repos import (
            SqlGeoPresenceRepository,
            SqlIncentiveRepository,
            SqlRateLimitRepository,
            SqlReputationRepository,
        )

        _ha_reputation_repo = SqlReputationRepository()
        _ha_incentive_repo = SqlIncentiveRepository()
        _ha_geo_repo = SqlGeoPresenceRepository()
        _ha_rate_repo = SqlRateLimitRepository()
        logger.info(
            "Shared HA state enabled (SQL reputation/incentives/geo/rate-limits)",
            extra={"component": "coordinator", "event": "ha_shared_state_on"},
        )
except Exception as _ha_exc:  # noqa: BLE001
    logger.warning("HA shared state init failed: %s", _ha_exc)

rate_limiter = RateLimiter(repo=_ha_rate_repo)
privacy_protector = PrivacyProtector()
update_validator = UpdateValidator(
    round_manager,
    auth_manager=auth_manager,
    rate_limiter=rate_limiter,
    privacy_protector=privacy_protector
)
metrics_collector = MetricsCollector()

# Async auto-aggregation (default ON for volunteer/edge)
enable_async = os.getenv("ENABLE_ASYNC_ROUNDS", "true").lower() == "true"
_aggregate_lock = threading.Lock()


def _auto_aggregate_round(round_id: int) -> None:
    """Callback: aggregate a round when min updates or timeout is reached."""
    with _aggregate_lock:
        round_obj = round_manager.rounds.get(round_id)
        if not round_obj or round_obj.state.value in ("AGGREGATING", "CLOSED"):
            return
        logger.info(
            f"Auto-aggregating round {round_id}",
            extra={
                "component": "coordinator",
                "event": "auto_aggregation_triggered",
                "round_id": round_id,
            },
        )
        result = aggregator.aggregate(round_id)
        if result and async_round_manager:
            async_round_manager.mark_round_closed(round_id)


async_config = AsyncRoundConfig(
    minimum_updates_required=int(os.getenv("ASYNC_MIN_UPDATES", "2")),
    max_round_duration_seconds=float(os.getenv("ASYNC_MAX_DURATION", "300.0")),
    enable_async=enable_async
)
async_round_manager = (
    AsyncRoundManager(
        round_manager,
        async_config,
        on_round_ready=_auto_aggregate_round,
    )
    if enable_async
    else None
)

aggregator = Aggregator(
    round_manager,
    model_store,
    task_assigner,
    metrics_collector,
    rate_limiter,
    state_store=state_store,
    on_aggregated=(
        (lambda rid: async_round_manager.mark_round_closed(rid))
        if async_round_manager
        else None
    ),
)

# Finish rounds interrupted mid-aggregate (Milestone 3)
_reconcile_results = aggregator.reconcile_after_restart()
if _reconcile_results:
    logger.info(
        f"Reconciled {len(_reconcile_results)} round(s) after restart",
        extra={
            "component": "coordinator",
            "event": "rounds_reconciled",
            "count": len(_reconcile_results),
        },
    )

reputation_manager = ReputationManager(repo=_ha_reputation_repo)
incentive_manager = IncentiveManager(
    base_reward_per_update=float(os.getenv("INCENTIVE_BASE_REWARD", "10.0")),
    speed_bonus_threshold=float(os.getenv("INCENTIVE_SPEED_THRESHOLD", "30.0")),
    consistency_bonus_threshold=int(os.getenv("INCENTIVE_CONSISTENCY_THRESHOLD", "5")),
    repo=_ha_incentive_repo,
)

# LoRA modules
base_model_registry = BaseModelRegistry()
lora_round_manager = get_lora_round_manager()
adapter_store = ModelStore(models_dir=str(Path(__file__).parent.parent.parent / "adapters"))
job_queue = get_job_queue()
local_launcher = get_local_launcher()
geo_presence = get_geo_presence(repo=_ha_geo_repo)


def _client_ip(http_request: Request) -> Optional[str]:
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else None


from protocol.routes import bind_dependencies, router as protocol_router
from protocol.credentials import extract_api_key
from protocol.version import ProtocolIncompatibleError, negotiate_protocol_version

bind_dependencies(
    auth_manager=auth_manager,
    round_manager=round_manager,
    update_validator=update_validator,
    metrics_collector=metrics_collector,
    geo_presence=geo_presence,
    client_ip_fn=_client_ip,
    register_legacy_fn=None,
)
app.include_router(protocol_router)

# Built-in classic FL architectures advertised to clients/UI
CLASSIC_MODEL_CATALOG = {
    "simple_mlp": {
        "description": "Built-in MLP (default federated trainer)",
        "config_schema": {
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01,
            "input_dim": 10,
            "hidden_dim": 32,
            "output_dim": 1,
        },
    },
    "tiny_cnn": {
        "description": "Built-in tiny CNN for image-like tensors",
        "config_schema": {
            "channels": 1,
            "image_size": 8,
            "num_classes": 2,
            "num_epochs": 2,
            "learning_rate": 0.01,
        },
    },
    "custom": {
        "description": "Load Trainer from client MODEL_MODULE=pkg.mod:Class",
        "config_schema": {},
    },
}


def operator_key_value(
    operator_key: Optional[str] = Query(None),
    x_operator_key: Optional[str] = Header(None, alias="X-Operator-Key"),
) -> Optional[str]:
    """Operator credential from header (preferred) or legacy query param."""
    return x_operator_key or operator_key


def _require_operator(operator_key: Optional[str]) -> None:
    if not validate_operator_key(operator_key):
        raise HTTPException(
            status_code=401,
            detail=(
                "Operator authentication required. Send the X-Operator-Key "
                "header (or legacy operator_key query parameter)."
            ),
        )


def _require_json_size(value: Any, *, env_name: str, default: int, label: str) -> None:
    limit = int(os.getenv(env_name, str(default)))
    size = len(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"{label} is {size} bytes; maximum is {limit}",
        )


# Pydantic models for request/response
class ClientRegisterRequest(BaseModel):
    """Request model for client registration."""
    client_name: str
    api_key: Optional[str] = None  # proof of possession when re-registering
    public_key: Optional[str] = None  # optional Ed25519 public key (Protocol V2)


class ClientRegisterResponse(BaseModel):
    """Response model for client registration."""
    success: bool
    message: str
    client_id: str
    api_key: str  # API key for authentication


class TaskResponse(BaseModel):
    """Response model for task assignment."""
    model_config = ConfigDict(populate_by_name=True)

    round_id: int
    model_version: str  # Changed to string format: "v1", "v2", etc.
    task: str
    description: str
    model_id: str = "simple_mlp"
    architecture_config: Dict[str, Any] = Field(default_factory=dict, alias="model_config")


class SetModelRequest(BaseModel):
    """Select architecture for classic FL rounds."""
    model_config = ConfigDict(populate_by_name=True)

    model_id: str
    architecture_config: Optional[Dict[str, Any]] = Field(default=None, alias="model_config")


class CreateJobRequest(BaseModel):
    """Create a general (non-training) job."""
    job_type: str  # inference | label | compute
    payload: Optional[Dict[str, Any]] = None
    priority: int = 0
    tags: Optional[list] = None
    max_attempts: int = 3
    lease_seconds: Optional[float] = None


class LaunchRequest(BaseModel):
    """Start local train clients or job workers from the ops UI."""
    kind: str  # train | worker
    count: int = 1
    model_id: Optional[str] = None
    model_module: Optional[str] = None
    dataset_preset: str = "none"
    dataset_path: Optional[str] = None
    job_types: str = "inference,label,compute"
    client_name_prefix: Optional[str] = None
    set_active_model: bool = True
    enqueue_sample_job: bool = False


class LaunchDemoRequest(BaseModel):
    """One-click local demo: model + clients + worker + sample job."""
    model_id: str = "simple_mlp"
    dataset_preset: str = "sample_private"
    train_clients: int = 2
    start_worker: bool = True
    enqueue_sample_job: bool = True


class JobResultRequest(BaseModel):
    """Submit result for a claimed job."""
    client_id: str
    api_key: Optional[str] = None
    result: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class UpdateRequest(BaseModel):
    """Request model for client update submission."""
    client_id: str
    round_id: int
    weight_delta: str
    api_key: Optional[str] = None  # API key for authentication


class UpdateResponse(BaseModel):
    """Response model for update submission."""
    success: bool
    message: str


class AggregateResponse(BaseModel):
    """Response model for aggregation."""
    round_id: int
    model_version: str  # New model version created after aggregation
    status: str
    aggregated_model: Optional[Dict[str, Any]]
    num_updates: int


class RoundStatusResponse(BaseModel):
    """Response model for round status."""
    round_id: int
    model_version: str  # Model version used for this round
    state: str
    assigned_clients: list
    updates_received: list
    total_clients: int
    total_updates: int


class ModelResponse(BaseModel):
    """Response model for model download."""
    version: str
    model_data: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    metrics: Dict[str, Any]


# LoRA-specific Pydantic models
class CreateLoRARoundRequest(BaseModel):
    """Request model for creating a LoRA fine-tuning round."""
    base_model_id: str
    adapter_version: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Optional[list[str]] = None
    max_steps: int = 100
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_seq_length: int = 512
    task_type: str = "causal_lm"


class LoRARoundResponse(BaseModel):
    """Response model for LoRA round configuration."""
    round_id: int
    base_model_id: str
    adapter_version: Optional[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_seq_length: int
    state: str
    created_at: str


class SubmitAdapterRequest(BaseModel):
    """Request model for submitting LoRA adapter."""
    client_id: str
    round_id: int
    adapter_state_dict: Dict[str, Any]
    num_samples: int
    training_loss: float
    api_key: Optional[str] = None


class SubmitAdapterResponse(BaseModel):
    """Response model for adapter submission."""
    success: bool
    message: str


class AggregateRoundRequest(BaseModel):
    """Request model for aggregating a round."""
    round_id: int
    weight_by_samples: bool = True
    strategy: Optional[str] = None  # delta_svd (default) | param_fedavg


class AggregateRoundResponse(BaseModel):
    """Response model for round aggregation."""
    round_id: int
    adapter_version: str
    status: str
    num_adapters: int
    evaluation_passed: bool
    evaluation_loss: Optional[float] = None
    aggregation_strategy: Optional[str] = None
    replayed: bool = False


@app.post("/client/register", response_model=ClientRegisterResponse)
async def register_client(
    request: ClientRegisterRequest,
    http_request: Request,
) -> ClientRegisterResponse:
    """
    Register a client and receive an API key.

    Idempotent only with proof of possession: returning clients must present
    their existing ``api_key``. Unauthenticated callers never receive an
    existing key (HTTP 409 if the name is taken).
    """
    geo_presence.record(request.client_name, _client_ip(http_request))
    logger.info(f"Registration request received for client {request.client_name}", extra={
        "component": "coordinator",
        "event": "registration_request",
        "client_id": request.client_name
    })

    already = auth_manager.is_registered(request.client_name)
    try:
        api_key = auth_manager.register_client(
            request.client_name,
            presented_key=request.api_key,
        )
    except ClientAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    if request.client_name not in round_manager.clients:
        round_manager.register_client(request.client_name)

    metrics_collector.total_clients_seen.add(request.client_name)

    if request.public_key:
        try:
            auth_manager.set_public_key(request.client_name, request.public_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    message = (
        f"Client {request.client_name} resumed with existing API key."
        if already
        else f"Client {request.client_name} registered successfully. Save your API key!"
    )
    logger.info(f"Client {request.client_name} registration ok", extra={
        "component": "coordinator",
        "event": "client_registered",
        "client_id": request.client_name
    })

    return ClientRegisterResponse(
        success=True,
        message=message,
        client_id=request.client_name,
        api_key=api_key
    )


@app.get("/task/{client_id}", response_model=TaskResponse)
async def get_task(
    client_id: str,
    http_request: Request,
    api_key: Optional[str] = Query(None, alias="api_key"),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> TaskResponse:
    """
    Get a task assignment for a client.
    
    Args:
        client_id: Identifier of the client requesting a task
        api_key: API key for authentication (``X-Api-Key`` preferred; query legacy)
        
    Returns:
        Task assignment with round_id, model_version, and task details
    """
    if x_protocol_version:
        try:
            negotiate_protocol_version(x_protocol_version)
        except ProtocolIncompatibleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_key = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        query_api_key=api_key,
    )
    # Authentication check
    if auth_manager and not auth_manager.validate_api_key(resolved_key, client_id):
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Valid API key required."
        )
    
    # Rate limiting check
    if rate_limiter:
        allowed, reason = rate_limiter.check_request_rate(client_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {reason}"
            )
        rate_limiter.record_request(client_id)

    geo_presence.record(client_id, _client_ip(http_request))
    task = task_assigner.assign_task(client_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not assign task to client {client_id}. Client may not be registered or already has an active assignment."
        )
    
    # Record client assignment in metrics
    round_id = task["round_id"]
    if async_round_manager:
        async_round_manager.start_round(round_id)
    round_status = round_manager.get_round_status(round_id)
    if round_status:
        # Check if this is a new round (need to start metrics tracking)
        if round_id not in metrics_collector.round_metrics:
            metrics_collector.start_round(round_id, task["model_version"])
        metrics_collector.record_client_assigned(round_id, client_id)
    
    return TaskResponse(
        round_id=task["round_id"],
        model_version=task["model_version"],
        task=task["task"],
        description=task["description"],
        model_id=task.get("model_id", "simple_mlp"),
        model_config=task.get("model_config") or {},
    )


@app.post("/update", response_model=UpdateResponse)
async def submit_update(
    request: UpdateRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> UpdateResponse:
    """
    Submit a client update.
    
    Args:
        request: Update request with client_id, round_id, weight_delta, and api_key
        
    Returns:
        Update submission response with success status
    """
    if x_protocol_version:
        try:
            negotiate_protocol_version(x_protocol_version)
        except ProtocolIncompatibleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_key = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        body_api_key=request.api_key,
    )

    # Check if round is closed (straggler detection)
    if async_round_manager and request.round_id in async_round_manager.closed_rounds:
        # This is a straggler - update arrived after round closed
        async_round_manager.record_straggler(request.client_id, request.round_id)
        reputation_manager.record_round_dropout(request.client_id, request.round_id)
        incentive_manager.record_dropout(request.client_id)
        raise HTTPException(
            status_code=410,  # Gone - round already closed
            detail=f"Round {request.round_id} is already closed. Update arrived too late."
        )
    
    # Validate update (includes authentication, rate limiting, value checks)
    is_valid, reason = update_validator.validate(
        request.client_id,
        request.round_id,
        request.weight_delta,
        api_key=resolved_key,
    )

    if is_valid:
        try:
            parsed_delta = json.loads(request.weight_delta)
        except (json.JSONDecodeError, TypeError):
            parsed_delta = request.weight_delta
        _require_json_size(
            parsed_delta,
            env_name="MAX_UPDATE_BYTES",
            default=25_000_000,
            label="Update payload",
        )
    
    if not is_valid:
        # Record rejected update in metrics and reputation
        metrics_collector.record_update_rejected(request.round_id)
        reputation_manager.record_update_rejected(request.client_id, request.round_id)
        
        # Provide specific error message
        if reason == "authentication_failed":
            status_code = 401
            detail = "Authentication failed. Valid API key required."
        elif reason == "rate_limit_exceeded":
            status_code = 429
            detail = f"Rate limit exceeded for client {request.client_id}"
        else:
            status_code = 400
            detail = f"Invalid update from client {request.client_id} for round {request.round_id}: {reason}"
        
        raise HTTPException(status_code=status_code, detail=detail)
    
    # Apply privacy protections
    protected_weight_delta = privacy_protector.protect_update(request.weight_delta)
    
    # Record rate limit usage
    if rate_limiter:
        rate_limiter.record_update(request.client_id, request.round_id)
    
    # Calculate latency for reputation and incentives
    latency = None
    if async_round_manager and request.round_id in async_round_manager.round_start_times:
        latency = time.time() - async_round_manager.round_start_times[request.round_id]
    
    # Submit update to aggregator
    success = aggregator.submit_update(
        request.client_id,
        request.round_id,
        protected_weight_delta
    )
    
    if not success:
        metrics_collector.record_update_rejected(request.round_id)
        reputation_manager.record_update_rejected(request.client_id, request.round_id)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to submit update from client {request.client_id} for round {request.round_id}"
        )
    
    # Record accepted update in metrics
    metrics_collector.record_update_received(request.round_id)
    metrics_collector.record_update_accepted(request.round_id)
    
    # Record in reputation system
    reputation_manager.record_update_submitted(request.client_id, request.round_id)
    reputation_manager.record_update_accepted(request.client_id, request.round_id)
    
    # Award incentives
    tokens_earned = incentive_manager.award_update_reward(
        request.client_id,
        request.round_id,
        latency_seconds=latency
    )
    
    # Check if round is ready for aggregation (async mode)
    if async_round_manager and async_round_manager.check_round_ready(request.round_id):
        # Round is ready - trigger aggregation callback
        if async_round_manager.on_round_ready:
            async_round_manager.on_round_ready(request.round_id)
    
    return UpdateResponse(
        success=True,
        message=f"Update from client {request.client_id} submitted successfully for round {request.round_id}"
    )


@app.get("/aggregate/{round_id}", response_model=AggregateResponse)
async def aggregate_classic_round(
    round_id: int,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> AggregateResponse:
    """
    Aggregate all updates for a classic FL round (FedAvg).

    When OPERATOR_API_KEY is set, operator_key is required.
    """
    _require_operator(operator_key)
    result = aggregator.aggregate(round_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )

    if async_round_manager and result.get("status") == "aggregated":
        async_round_manager.mark_round_closed(round_id)

    return AggregateResponse(
        round_id=result["round_id"],
        model_version=result.get("model_version") or "",
        status=result["status"],
        aggregated_model=result.get("aggregated_model"),
        num_updates=result.get("num_updates", 0),
    )


@app.get("/status/{round_id}", response_model=RoundStatusResponse)
async def get_round_status(round_id: int) -> RoundStatusResponse:
    """
    Get the status of a round.
    
    Args:
        round_id: Identifier of the round
        
    Returns:
        Round status information
    """
    status = round_manager.get_round_status(round_id)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    
    return RoundStatusResponse(
        round_id=status["round_id"],
        model_version=status["model_version"],
        state=status["state"],
        assigned_clients=status["assigned_clients"],
        updates_received=status["updates_received"],
        total_clients=status["total_clients"],
        total_updates=status["total_updates"]
    )


@app.get("/model/{version}", response_model=ModelResponse)
async def get_model(version: str) -> ModelResponse:
    """
    Get a specific model version.
    
    Args:
        version: Model version string (e.g., "v1", "v2")
        
    Returns:
        Model data for the specified version
        
    Raises:
        HTTPException: 404 if model version does not exist
    """
    try:
        model_data = model_store.load_model(version)
        return ModelResponse(
            version=version,
            model_data=model_data
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {version} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {version}: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_all_metrics() -> MetricsResponse:
    """
    Get all metrics (global and per-round).
    
    Returns:
        All metrics including global statistics and round-specific metrics
    """
    return MetricsResponse(metrics=metrics_collector.get_all_metrics())


@app.get("/metrics/latest", response_model=Dict[str, Any])
async def get_latest_metrics() -> Dict[str, Any]:
    """
    Get metrics for the most recent round.
    
    Returns:
        Latest round metrics, or empty dict if no rounds exist
    """
    latest = metrics_collector.get_latest_round_metrics()
    if latest is None:
        return {}
    return latest


@app.get("/metrics/round/{round_id}", response_model=Dict[str, Any])
async def get_round_metrics(round_id: int) -> Dict[str, Any]:
    """
    Get metrics for a specific round.
    
    Args:
        round_id: Identifier of the round
        
    Returns:
        Round metrics, or empty dict if round not found
        
    Raises:
        HTTPException: 404 if round not found
    """
    metrics = metrics_collector.get_round_metrics(round_id)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for round {round_id} not found"
        )
    return metrics


@app.get("/reputation", response_model=Dict[str, Any])
async def get_all_reputations() -> Dict[str, Any]:
    """
    Get all client reputations.
    
    Returns:
        Dictionary mapping client_id to reputation data
    """
    return reputation_manager.get_all_reputations()


@app.get("/reputation/{client_id}", response_model=Dict[str, Any])
async def get_client_reputation(client_id: str) -> Dict[str, Any]:
    """
    Get reputation for a specific client.
    
    Args:
        client_id: Identifier of the client
        
    Returns:
        Client reputation data
        
    Raises:
        HTTPException: 404 if client not found
    """
    rep = reputation_manager.get_reputation(client_id)
    if rep is None:
        raise HTTPException(
            status_code=404,
            detail=f"Reputation for client {client_id} not found"
        )
    return rep.to_dict()


@app.get("/incentives", response_model=Dict[str, Any])
async def get_all_incentives() -> Dict[str, Any]:
    """
    Get all client incentives.
    
    Returns:
        Dictionary mapping client_id to incentive data
    """
    return incentive_manager.get_all_incentives()


@app.get("/incentives/{client_id}", response_model=Dict[str, Any])
async def get_client_incentives(client_id: str) -> Dict[str, Any]:
    """
    Get incentives for a specific client.
    
    Args:
        client_id: Identifier of the client
        
    Returns:
        Client incentive data
        
    Raises:
        HTTPException: 404 if client not found
    """
    incentives = incentive_manager.get_client_incentives(client_id)
    if incentives is None:
        raise HTTPException(
            status_code=404,
            detail=f"Incentives for client {client_id} not found"
        )
    return incentives.to_dict()


@app.get("/async/round/{round_id}/stats", response_model=Dict[str, Any])
async def get_async_round_stats(round_id: int) -> Dict[str, Any]:
    """
    Get async round statistics.
    
    Args:
        round_id: Identifier of the round
        
    Returns:
        Async round statistics
        
    Raises:
        HTTPException: 404 if async mode not enabled or round not found
    """
    if not async_round_manager:
        raise HTTPException(
            status_code=404,
            detail="Async round management is not enabled"
        )
    
    stats = async_round_manager.get_round_stats(round_id)
    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    return stats


# LoRA Fine-Tuning Endpoints

@app.get("/dashboard/activity")
async def dashboard_activity() -> Dict[str, Any]:
    """
    Anonymized map of recently active clients for the public landing page.

    Locations are city-level with per-client jitter; no IDs or IPs are
    exposed. Nodes seen in the last 5 minutes are flagged online.
    """
    nodes = geo_presence.snapshot()
    return {
        "server_time": time.time(),
        "nodes": nodes,
        "online_count": sum(1 for n in nodes if n["online"]),
    }


@app.get("/dashboard/overview")
async def dashboard_overview(
    limit: int = Query(25, ge=1, le=100),
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """
    Aggregated payload for the ops UI.

    Returns health, global metrics, recent classic rounds, reputations,
    incentives, LoRA base models, and recent LoRA rounds.
    Job payload/result bodies are redacted unless operator auth succeeds
    when OPERATOR_API_KEY is configured.
    """
    include_sensitive = True
    if get_operator_api_key() is not None:
        include_sensitive = validate_operator_key(operator_key)
    all_metrics = metrics_collector.get_all_metrics()
    global_metrics = all_metrics.get("global", {})
    round_metrics_map = all_metrics.get("rounds", {})

    # Prefer live RoundManager state; fall back to metrics keys
    classic_ids = sorted(round_manager.rounds.keys(), reverse=True)
    if not classic_ids:
        classic_ids = sorted(
            (int(k) for k in round_metrics_map.keys()),
            reverse=True
        )
    classic_ids = classic_ids[:limit]

    classic_rounds: List[Dict[str, Any]] = []
    for round_id in classic_ids:
        status = round_manager.get_round_status(round_id)
        metrics = round_metrics_map.get(str(round_id)) or round_metrics_map.get(round_id) or {}
        if status is None:
            classic_rounds.append({
                "round_id": round_id,
                "model_version": metrics.get("model_version"),
                "state": "UNKNOWN",
                "assigned_clients": [],
                "updates_received": [],
                "total_clients": metrics.get("clients_assigned", 0),
                "total_updates": metrics.get("updates_received", 0),
                "metrics": metrics,
            })
        else:
            classic_rounds.append({**status, "metrics": metrics})

    latest = metrics_collector.get_latest_round_metrics()

    return {
        "version": "1.0.0",
        "async_enabled": enable_async,
        "operator_auth_required": bool(get_operator_api_key()),
        "server_time": time.time(),
        "global": global_metrics,
        "latest_round": latest or {},
        "classic_rounds": classic_rounds,
        "reputations": reputation_manager.get_all_reputations(),
        "incentives": incentive_manager.get_all_incentives(),
        "lora_base_models": base_model_registry.list_models(),
        "lora_adapters": adapter_store.list_models(),
        "lora_rounds": lora_round_manager.list_rounds(limit=limit),
        "registered_clients": sorted(list(round_manager.clients)),
        "jobs": job_queue.list_jobs(
            limit=min(limit, 25),
            include_sensitive=include_sensitive,
        ),
        "job_stats": job_queue.stats(),
        "classic_models": CLASSIC_MODEL_CATALOG,
        "active_model": {
            "model_id": task_assigner.model_id,
            "model_config": task_assigner.model_config,
        },
        "launcher": local_launcher.status(),
    }


@app.post("/rounds/create", response_model=LoRARoundResponse)
async def create_round(
    request: CreateLoRARoundRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> LoRARoundResponse:
    """
    Create a new LoRA fine-tuning round.
    
    Args:
        request: Round configuration
        
    Returns:
        LoRARoundResponse with round details
    """
    _require_operator(operator_key)
    # Validate base model exists
    if not base_model_registry.model_exists(request.base_model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Base model {request.base_model_id} not found. Available: {base_model_registry.list_models()}"
        )
    if request.adapter_version:
        if not adapter_store.model_exists(request.adapter_version):
            raise HTTPException(
                status_code=400,
                detail=f"Adapter {request.adapter_version} not found",
            )
        previous = adapter_store.load_model(request.adapter_version)
        if previous.get("base_model_id") != request.base_model_id:
            raise HTTPException(
                status_code=400,
                detail="Previous adapter uses a different base model",
            )
        previous_lora = previous.get("lora_config") or {}
        requested_targets = request.target_modules or ["q_proj", "v_proj"]
        if previous_lora and (
            previous_lora.get("r") != request.lora_r
            or previous_lora.get("target_modules") != requested_targets
        ):
            raise HTTPException(
                status_code=400,
                detail="Previous adapter has incompatible rank or target modules",
            )
    
    # Create round
    config = create_lora_round(
        base_model_id=request.base_model_id,
        adapter_version=request.adapter_version,
        lora_r=request.lora_r,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
        target_modules=request.target_modules,
        max_steps=request.max_steps,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        warmup_steps=request.warmup_steps,
        max_seq_length=request.max_seq_length,
        task_type=request.task_type,
    )
    
    return LoRARoundResponse(
        round_id=config.round_id,
        base_model_id=config.base_model_id,
        adapter_version=config.adapter_version,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_seq_length=config.max_seq_length,
        state=config.state,
        created_at=config.created_at
    )


@app.get("/rounds/{round_id}", response_model=LoRARoundResponse)
async def get_round(
    round_id: int,
    api_key: Optional[str] = Query(None, alias="api_key")
) -> LoRARoundResponse:
    """
    Get LoRA round configuration.
    
    Args:
        round_id: Round identifier
        api_key: Optional API key for authentication
        
    Returns:
        LoRARoundResponse with round configuration
    """
    config = get_lora_round(round_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    
    return LoRARoundResponse(
        round_id=config.round_id,
        base_model_id=config.base_model_id,
        adapter_version=config.adapter_version,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_seq_length=config.max_seq_length,
        state=config.state,
        created_at=config.created_at
    )


@app.post("/rounds/{round_id}/submit", response_model=SubmitAdapterResponse)
async def submit_adapter(
    round_id: int,
    request: SubmitAdapterRequest
) -> SubmitAdapterResponse:
    """
    Submit a LoRA adapter for a round.
    
    Args:
        round_id: Round identifier
        request: Adapter submission request
        
    Returns:
        SubmitAdapterResponse with submission status
    """
    # Authentication
    if not auth_manager.validate_api_key(request.api_key, request.client_id):
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Valid API key required."
        )
    _require_json_size(
        request.adapter_state_dict,
        env_name="MAX_ADAPTER_UPLOAD_BYTES",
        default=100_000_000,
        label="Adapter upload",
    )
    
    # Rate limiting
    if rate_limiter:
        allowed, reason = rate_limiter.check_update_rate(request.client_id, round_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {reason}"
            )
        rate_limiter.record_update(request.client_id, round_id)
    
    # Validate round exists
    config = get_lora_round(round_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    
    if config.state == "CLOSED":
        raise HTTPException(
            status_code=410,
            detail=f"Round {round_id} is already closed"
        )
    
    # Validate adapter
    is_valid, error_msg = validate_adapter(request.adapter_state_dict)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid adapter: {error_msg}"
        )
    
    # Compute adapter hash
    adapter_json = json.dumps(request.adapter_state_dict, sort_keys=True)
    adapter_hash = hashlib.sha256(adapter_json.encode()).hexdigest()[:16]
    
    # Submit adapter
    success = lora_round_manager.submit_adapter(
        round_id=round_id,
        client_id=request.client_id,
        adapter_state_dict=request.adapter_state_dict,
        num_samples=request.num_samples,
        training_loss=request.training_loss,
        adapter_hash=adapter_hash
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to submit adapter for round {round_id}"
        )
    
    # Record metrics
    metrics_collector.record_update_received(round_id)
    metrics_collector.record_update_accepted(round_id)
    reputation_manager.record_update_submitted(request.client_id, round_id)
    reputation_manager.record_update_accepted(request.client_id, round_id)
    
    return SubmitAdapterResponse(
        success=True,
        message=f"Adapter submitted successfully for round {round_id}"
    )


@app.get("/adapters/{version}")
async def download_adapter(
    version: str,
    client_id: str,
    api_key: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Download a prior aggregated adapter for continued LoRA training."""
    if not auth_manager.validate_api_key(api_key, client_id):
        raise HTTPException(status_code=401, detail="Authentication failed")
    if not adapter_store.model_exists(version):
        raise HTTPException(status_code=404, detail=f"Adapter {version} not found")
    return adapter_store.load_model(version)


@app.post("/rounds/{round_id}/aggregate", response_model=AggregateRoundResponse)
async def aggregate_lora_round(
    round_id: int,
    request: AggregateRoundRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> AggregateRoundResponse:
    """
    Aggregate LoRA adapters for a round (default ΔW+SVD FedAvg).
    """
    _require_operator(operator_key)
    config = get_lora_round(round_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )

    # Idempotent re-aggregate: return previously published adapter
    if config.state == "CLOSED" and config.published_version:
        previous = (
            adapter_store.load_model(config.published_version)
            if adapter_store.model_exists(config.published_version)
            else {}
        )
        return AggregateRoundResponse(
            round_id=round_id,
            adapter_version=config.published_version,
            status="already_closed",
            num_adapters=len(lora_round_manager.get_submissions(round_id) or {}),
            evaluation_passed=bool((previous.get("evaluation") or {}).get("passed")),
            evaluation_loss=(previous.get("evaluation") or {}).get("loss"),
            aggregation_strategy=config.aggregation_strategy
            or (previous.get("manifest") or {}).get("aggregation_strategy"),
            replayed=True,
        )

    if config.state == "CLOSED":
        raise HTTPException(
            status_code=400,
            detail=f"Round {round_id} is already closed"
        )

    submissions = lora_round_manager.get_submissions(round_id)
    if not submissions:
        raise HTTPException(
            status_code=400,
            detail=f"No adapter submissions for round {round_id}"
        )
    strategy = request.strategy or os.getenv("LORA_AGG_STRATEGY", "delta_svd")
    lora_round_manager.set_state(round_id, "AGGREGATING")

    aggregated_adapter = aggregate_lora_adapters(
        submissions,
        weight_by_samples=request.weight_by_samples,
        strategy=strategy,
    )

    if aggregated_adapter is None:
        lora_round_manager.set_state(round_id, "COLLECTING")
        raise HTTPException(
            status_code=500,
            detail="Failed to aggregate adapters"
        )

    version_base = adapter_store.latest_model_version()
    adapter_version = next_version(version_base) if version_base else "v1"

    previous_loss = None
    if config.adapter_version and adapter_store.model_exists(config.adapter_version):
        previous_adapter = adapter_store.load_model(config.adapter_version)
        previous_loss = (
            previous_adapter.get("evaluation", {}).get("loss")
        )
    base_model = base_model_registry.get_model_config(config.base_model_id)
    if base_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown base model: {config.base_model_id}",
        )

    lora_round_manager.set_state(round_id, "EVALUATING")
    try:
        eval_result = evaluate_adapter(
            round_id=round_id,
            adapter_version=adapter_version,
            aggregated_adapter=aggregated_adapter,
            base_model_name=base_model.model_name,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            max_seq_length=config.max_seq_length,
            previous_adapter_loss=previous_loss,
            task_type=getattr(config, "task_type", None) or "causal_lm",
        )
    except Exception as exc:
        lora_round_manager.set_state(round_id, "COLLECTING")
        raise HTTPException(
            status_code=500,
            detail=f"LoRA evaluation failed: {exc}",
        ) from exc
    if (
        eval_result.evaluated
        and not eval_result.passed
        and os.getenv("LORA_REJECT_REGRESSION", "false").lower()
        in {"1", "true", "yes"}
    ):
        lora_round_manager.set_state(round_id, "REJECTED")
        raise HTTPException(
            status_code=422,
            detail="Aggregated adapter regressed on the holdout dataset",
        )

    manifest = build_adapter_manifest(
        adapter_version=adapter_version,
        adapter_state_dict=aggregated_adapter,
        base_model_id=config.base_model_id,
        round_id=round_id,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        aggregation_strategy=strategy,
        task_type=getattr(config, "task_type", "causal_lm"),
        storage_uri=f"file://adapters/model_{adapter_version}.json",
        extra_metadata={"num_clients": len(submissions)},
    )
    register_adapter_manifest(manifest)

    adapter_data = {
        "version": adapter_version,
        "round_id": round_id,
        "base_model_id": config.base_model_id,
        "lora_config": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.target_modules,
        },
        "adapter_state_dict": aggregated_adapter,
        "num_clients": len(submissions),
        "aggregation_strategy": strategy,
        "task_type": getattr(config, "task_type", "causal_lm"),
        "manifest": manifest.to_dict(),
        "evaluation": {
            "loss": eval_result.evaluation_loss,
            "passed": eval_result.passed,
            "evaluated": eval_result.evaluated,
            "num_samples": eval_result.num_eval_samples,
            "reason": eval_result.reason,
            "task_type": eval_result.task_type,
            "metrics": eval_result.metrics,
        },
        "created_at": time.time()
    }

    try:
        adapter_store.save_model(adapter_version, adapter_data)
    except Exception as e:
        lora_round_manager.set_state(round_id, "COLLECTING")
        logger.error(f"Failed to save adapter {adapter_version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save aggregated adapter: {e}"
        )

    config.published_version = adapter_version
    config.aggregation_strategy = strategy
    close_lora_round(round_id)

    return AggregateRoundResponse(
        round_id=round_id,
        adapter_version=adapter_version,
        status="aggregated",
        num_adapters=len(submissions),
        evaluation_passed=eval_result.passed,
        evaluation_loss=eval_result.evaluation_loss,
        aggregation_strategy=strategy,
        replayed=False,
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness/readiness for edge orchestration and Docker healthchecks."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "async_enabled": enable_async,
        "registered_clients": len(round_manager.clients),
        "operator_auth_required": get_operator_api_key() is not None,
        "job_queue": job_queue.stats(),
        "active_model_id": task_assigner.model_id,
    }


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    """
    Readiness for multi-replica / k8s probes.

    Checks metadata backend configuration and artifact store reachability.
    Liveness remains GET /health.
    """
    from persistence.json_repos import metadata_backend
    from artifacts import get_artifact_store

    backend = metadata_backend()
    checks: Dict[str, Any] = {
        "metadata_backend": backend,
        "artifact_store": os.getenv("ARTIFACT_STORE", "local"),
        "operator_auth_required": get_operator_api_key() is not None,
    }
    try:
        if backend in ("postgres", "sqlite", "sql"):
            from persistence.db import get_engine
            from sqlalchemy import text

            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        else:
            checks["database"] = "skipped"
        store = get_artifact_store()
        # Local: root exists; S3: client constructs (head optional probe object)
        if hasattr(store, "root"):
            checks["artifacts"] = "ok" if store.root.exists() else "missing_root"
        else:
            _ = store.client  # ensure boto3 client builds
            checks["artifacts"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(exc), **checks})
    return {"status": "ready", **checks}


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List classic FL architectures and LoRA base models."""
    return {
        "classic": CLASSIC_MODEL_CATALOG,
        "active_classic": {
            "model_id": task_assigner.model_id,
            "model_config": task_assigner.model_config,
            "model_version": task_assigner.model_version,
        },
        "lora_base_models": base_model_registry.list_models(),
    }


@app.post("/models/active")
async def set_active_model(
    request: SetModelRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """Select which classic architecture new FL rounds use."""
    _require_operator(operator_key)
    if request.model_id not in CLASSIC_MODEL_CATALOG and request.model_id != "custom":
        # Allow unknown ids for custom plugins advertised by operators
        pass
    task_assigner.set_model(request.model_id, request.architecture_config)
    return {
        "success": True,
        "model_id": task_assigner.model_id,
        "model_config": task_assigner.model_config,
    }


@app.post("/jobs")
async def create_job(
    request: CreateJobRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """Enqueue a non-training job (inference | label | compute)."""
    _require_operator(operator_key)
    allowed = {"inference", "label", "compute"}
    if request.job_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"job_type must be one of {sorted(allowed)}",
        )
    _require_json_size(
        request.payload or {},
        env_name="MAX_JOB_PAYLOAD_BYTES",
        default=1_000_000,
        label="Job payload",
    )
    try:
        job = job_queue.create_job(
            job_type=request.job_type,
            payload=request.payload,
            priority=request.priority,
            tags=request.tags,
            max_attempts=request.max_attempts,
            lease_seconds=request.lease_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@app.get("/jobs")
async def list_jobs(
    state: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """List jobs. Payload/result bodies require operator auth when OPERATOR_API_KEY is set."""
    include_sensitive = True
    if get_operator_api_key() is not None:
        include_sensitive = validate_operator_key(operator_key)
    return {
        "jobs": job_queue.list_jobs(
            state=state,
            job_type=job_type,
            limit=limit,
            include_sensitive=include_sensitive,
        ),
        "stats": job_queue.stats(),
    }


@app.get("/jobs/claim")
async def claim_job(
    client_id: str,
    api_key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    types: Optional[str] = Query(
        None,
        description="Comma-separated job types, e.g. inference,compute",
    ),
) -> Dict[str, Any]:
    """Claim the next queued job for a volunteer/edge client."""
    resolved = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        query_api_key=api_key,
    )
    if not auth_manager.validate_api_key(resolved, client_id):
        raise HTTPException(status_code=401, detail="Authentication failed")
    type_set = None
    if types:
        type_set = {t.strip() for t in types.split(",") if t.strip()}
    job = job_queue.claim_next(client_id, job_types=type_set)
    if job is None:
        return {"job": None, "message": "No jobs available"}
    return {
        "job": job.to_dict(include_sensitive=True),
        "lease_seconds": job.lease_seconds or job_queue.lease_seconds,
        "lease_expires_at": job.lease_expires_at,
    }


@app.post("/jobs/{job_id}/lease")
async def extend_job_lease(
    job_id: str,
    client_id: str = Query(...),
    api_key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    extend_seconds: Optional[float] = Query(None),
) -> Dict[str, Any]:
    """Heartbeat: extend the lease for a claimed job."""
    resolved = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        query_api_key=api_key,
    )
    if not auth_manager.validate_api_key(resolved, client_id):
        raise HTTPException(status_code=401, detail="Authentication failed")
    try:
        job = job_queue.extend_lease(
            job_id, client_id, extend_seconds=extend_seconds
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found or not assigned to this client",
        )
    return {
        "job_id": job.job_id,
        "lease_expires_at": job.lease_expires_at,
        "lease_seconds": job.lease_seconds or job_queue.lease_seconds,
    }


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    include_sensitive = True
    if get_operator_api_key() is not None:
        include_sensitive = validate_operator_key(operator_key)
    return job.to_dict(include_sensitive=include_sensitive)


@app.post("/jobs/{job_id}/result")
async def submit_job_result(job_id: str, request: JobResultRequest) -> Dict[str, Any]:
    """Submit results for a claimed job (private data stays on client)."""
    if not auth_manager.validate_api_key(request.api_key, request.client_id):
        raise HTTPException(status_code=401, detail="Authentication failed")
    _require_json_size(
        request.result,
        env_name="MAX_JOB_RESULT_BYTES",
        default=5_000_000,
        label="Job result",
    )
    job = job_queue.submit_result(
        job_id=job_id,
        client_id=request.client_id,
        result=request.result,
        success=request.success,
        error=request.error,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to client")
    return job.to_dict()


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """Cancel a queued, assigned, or failed job."""
    _require_operator(operator_key)
    job = job_queue.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or already final")
    return job.to_dict()


class DatasetAliasRequest(BaseModel):
    alias: str
    description: str = ""
    format_hint: Optional[str] = None
    required_env: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/datasets/aliases")
async def list_dataset_aliases(
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """List registered dataset aliases (no absolute paths)."""
    _require_operator(operator_key)
    from jobs import get_dataset_alias_registry

    return {"aliases": get_dataset_alias_registry().list_aliases()}


@app.post("/datasets/aliases")
async def upsert_dataset_alias(
    request: DatasetAliasRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """Register or update a dataset alias (workers map alias → local path)."""
    _require_operator(operator_key)
    from jobs import get_dataset_alias_registry

    try:
        record = get_dataset_alias_registry().upsert(
            request.alias,
            description=request.description,
            format_hint=request.format_hint,
            required_env=request.required_env,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.to_dict()


@app.get("/launch")
async def launch_status() -> Dict[str, Any]:
    """List UI-started local clients/workers."""
    return local_launcher.status()


@app.post("/launch")
async def launch_process(
    request: LaunchRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """Start local train clients or job workers (dev/ops UI)."""
    _require_operator(operator_key)
    if not local_launcher.enabled:
        raise HTTPException(
            status_code=403,
            detail="Local launcher disabled. Set ENABLE_LOCAL_LAUNCHER=true",
        )
    try:
        if request.set_active_model and request.kind == "train" and request.model_id:
            task_assigner.set_model(request.model_id, None)
        started = local_launcher.start(
            kind=request.kind,
            count=request.count,
            model_id=request.model_id,
            model_module=request.model_module,
            dataset_preset=request.dataset_preset,
            dataset_path=request.dataset_path,
            job_types=request.job_types,
            client_name_prefix=request.client_name_prefix,
        )
        sample_job = None
        if request.enqueue_sample_job and request.kind == "worker":
            sample_job = job_queue.create_job(
                "compute",
                {
                    "entrypoint": "examples.science_plugin:lennard_jones",
                    "work_unit": {
                        "positions": [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]],
                        "steps": 250,
                        "dt": 0.001,
                    },
                },
            ).to_dict()
        return {
            "success": True,
            "started": started,
            "launcher": local_launcher.status(),
            "sample_job": sample_job,
            "active_model": {
                "model_id": task_assigner.model_id,
                "model_config": task_assigner.model_config,
            },
        }
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/launch/demo")
async def launch_demo(
    request: LaunchDemoRequest,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    """One-click: set model, start train clients + worker, enqueue sample job."""
    _require_operator(operator_key)
    if not local_launcher.enabled:
        raise HTTPException(
            status_code=403,
            detail="Local launcher disabled. Set ENABLE_LOCAL_LAUNCHER=true",
        )
    try:
        task_assigner.set_model(request.model_id, None)
        started = []
        started.extend(
            local_launcher.start(
                kind="train",
                count=request.train_clients,
                model_id=request.model_id,
                dataset_preset=request.dataset_preset,
                client_name_prefix="ui-train",
            )
        )
        if request.start_worker:
            started.extend(
                local_launcher.start(
                    kind="worker",
                    count=1,
                    dataset_preset=request.dataset_preset,
                    job_types="inference,label,compute",
                    client_name_prefix="ui-worker",
                )
            )
        sample_job = None
        if request.enqueue_sample_job:
            sample_job = job_queue.create_job(
                "compute",
                {
                    "entrypoint": "examples.science_plugin:lennard_jones",
                    "work_unit": {
                        "positions": [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]],
                        "steps": 250,
                        "dt": 0.001,
                    },
                },
            ).to_dict()
        return {
            "success": True,
            "started": started,
            "sample_job": sample_job,
            "active_model": {"model_id": task_assigner.model_id},
            "launcher": local_launcher.status(),
        }
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/launch/{process_id}/stop")
async def stop_launch(
    process_id: str,
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    _require_operator(operator_key)
    ok = local_launcher.stop(process_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Process not found or already stopped")
    return {"success": True, "launcher": local_launcher.status()}


@app.post("/launch/stop-all")
async def stop_all_launch(
    kind: Optional[str] = Query(None),
    operator_key: Optional[str] = Depends(operator_key_value),
) -> Dict[str, Any]:
    _require_operator(operator_key)
    n = local_launcher.stop_all(kind=kind)
    return {"success": True, "stopped": n, "launcher": local_launcher.status()}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    endpoints = {
        "health": "GET /health",
        "ready": "GET /ready",
        "register_client": "POST /client/register",
        "get_task": "GET /task/{client_id}",
        "submit_update": "POST /update",
        "aggregate_round": "GET /aggregate/{round_id}",
        "get_round_status": "GET /status/{round_id}",
        "get_model": "GET /model/{version}",
        "list_models": "GET /models",
        "set_active_model": "POST /models/active",
        "create_job": "POST /jobs",
        "list_jobs": "GET /jobs",
        "claim_job": "GET /jobs/claim",
        "submit_job_result": "POST /jobs/{job_id}/result",
        "launch_status": "GET /launch",
        "launch_start": "POST /launch",
        "launch_demo": "POST /launch/demo",
        "launch_stop": "POST /launch/{process_id}/stop",
        "launch_stop_all": "POST /launch/stop-all",
        "get_all_metrics": "GET /metrics",
        "get_latest_metrics": "GET /metrics/latest",
        "get_round_metrics": "GET /metrics/round/{round_id}",
        "get_all_reputations": "GET /reputation",
        "get_client_reputation": "GET /reputation/{client_id}",
        "get_all_incentives": "GET /incentives",
        "get_client_incentives": "GET /incentives/{client_id}",
        "get_async_round_stats": "GET /async/round/{round_id}/stats",
        "dashboard_overview": "GET /dashboard/overview",
        "create_lora_round": "POST /rounds/create",
        "get_lora_round": "GET /rounds/{round_id}",
        "download_lora_adapter": "GET /adapters/{version}",
        "submit_lora_adapter": "POST /rounds/{round_id}/submit",
        "aggregate_lora_round": "POST /rounds/{round_id}/aggregate"
    }
    
    return {
        "message": "Federated Learning Coordinator API",
        "version": "1.1.0",
        "async_enabled": enable_async,
        "endpoints": endpoints
    }


@app.get("/dashboard")
async def dashboard_redirect():
    """Redirect to the ops UI when a production build is present."""
    return RedirectResponse(url="/ui/")


# Serve production UI build if present (ui/dist next to coordinator/)
_ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"
if _ui_dist.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui_dist), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

