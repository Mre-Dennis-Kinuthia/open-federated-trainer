"""
FastAPI Server for Federated Learning Coordinator

Main entry point for the coordinator API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path

from core.round_manager import RoundManager
from core.task_assigner import TaskAssigner
from core.update_validator import UpdateValidator
from core.aggregator import Aggregator
from core.model_store import ModelStore
from core.metrics import MetricsCollector
from core.auth import AuthManager, validate_operator_key, get_operator_api_key
from core.rate_limiter import RateLimiter
from core.privacy import PrivacyProtector
from core.async_round_manager import AsyncRoundManager, AsyncRoundConfig
from core.reputation import ReputationManager
from core.incentives import IncentiveManager
from core.state_store import StateStore
from model_registry.base_models import BaseModelRegistry
from rounds import create_lora_round, get_lora_round, close_lora_round
from rounds.create_round import get_lora_round_manager
from aggregation import aggregate_lora_adapters, validate_adapter
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
round_manager = RoundManager(state_store=state_store)
task_assigner = TaskAssigner(round_manager, model_store)
auth_manager = AuthManager(state_store=state_store)
rate_limiter = RateLimiter()
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

reputation_manager = ReputationManager()
incentive_manager = IncentiveManager(
    base_reward_per_update=float(os.getenv("INCENTIVE_BASE_REWARD", "10.0")),
    speed_bonus_threshold=float(os.getenv("INCENTIVE_SPEED_THRESHOLD", "30.0")),
    consistency_bonus_threshold=int(os.getenv("INCENTIVE_CONSISTENCY_THRESHOLD", "5"))
)

# LoRA modules
base_model_registry = BaseModelRegistry()
lora_round_manager = get_lora_round_manager()
adapter_store = ModelStore(models_dir=str(Path(__file__).parent.parent.parent / "adapters"))


def _require_operator(operator_key: Optional[str]) -> None:
    if not validate_operator_key(operator_key):
        raise HTTPException(
            status_code=401,
            detail="Operator authentication required. Set OPERATOR_API_KEY and pass operator_key.",
        )

# Pydantic models for request/response
class ClientRegisterRequest(BaseModel):
    """Request model for client registration."""
    client_name: str


class ClientRegisterResponse(BaseModel):
    """Response model for client registration."""
    success: bool
    message: str
    client_id: str
    api_key: str  # API key for authentication


class TaskResponse(BaseModel):
    """Response model for task assignment."""
    round_id: int
    model_version: str  # Changed to string format: "v1", "v2", etc.
    task: str
    description: str


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


class AggregateRoundResponse(BaseModel):
    """Response model for round aggregation."""
    round_id: int
    adapter_version: str
    status: str
    num_adapters: int
    evaluation_passed: bool
    evaluation_loss: Optional[float] = None


@app.post("/client/register", response_model=ClientRegisterResponse)
async def register_client(request: ClientRegisterRequest) -> ClientRegisterResponse:
    """
    Register a client and receive an API key.

    Idempotent for volunteer/edge: returning clients get their existing key.
    """
    logger.info(f"Registration request received for client {request.client_name}", extra={
        "component": "coordinator",
        "event": "registration_request",
        "client_id": request.client_name
    })

    already = request.client_name in round_manager.clients
    if not already:
        round_manager.register_client(request.client_name)

    api_key = auth_manager.register_client(request.client_name)
    metrics_collector.total_clients_seen.add(request.client_name)

    message = (
        f"Client {request.client_name} already registered; returning existing API key."
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
    api_key: Optional[str] = Query(None, alias="api_key")
) -> TaskResponse:
    """
    Get a task assignment for a client.
    
    Args:
        client_id: Identifier of the client requesting a task
        api_key: API key for authentication (query parameter or header)
        
    Returns:
        Task assignment with round_id, model_version, and task details
    """
    # Authentication check
    if auth_manager and not auth_manager.validate_api_key(api_key, client_id):
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
        description=task["description"]
    )


@app.post("/update", response_model=UpdateResponse)
async def submit_update(request: UpdateRequest) -> UpdateResponse:
    """
    Submit a client update.
    
    Args:
        request: Update request with client_id, round_id, weight_delta, and api_key
        
    Returns:
        Update submission response with success status
    """
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
        api_key=request.api_key
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
    operator_key: Optional[str] = Query(None),
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

@app.get("/dashboard/overview")
async def dashboard_overview(limit: int = Query(25, ge=1, le=100)) -> Dict[str, Any]:
    """
    Aggregated payload for the ops UI.

    Returns health, global metrics, recent classic rounds, reputations,
    incentives, LoRA base models, and recent LoRA rounds.
    """
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
        "global": global_metrics,
        "latest_round": latest or {},
        "classic_rounds": classic_rounds,
        "reputations": reputation_manager.get_all_reputations(),
        "incentives": incentive_manager.get_all_incentives(),
        "lora_base_models": base_model_registry.list_models(),
        "lora_rounds": lora_round_manager.list_rounds(limit=limit),
        "registered_clients": sorted(list(round_manager.clients)),
    }


@app.post("/rounds/create", response_model=LoRARoundResponse)
async def create_round(
    request: CreateLoRARoundRequest,
    operator_key: Optional[str] = Query(None),
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
        max_seq_length=request.max_seq_length
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


@app.post("/rounds/{round_id}/aggregate", response_model=AggregateRoundResponse)
async def aggregate_lora_round(
    round_id: int,
    request: AggregateRoundRequest,
    operator_key: Optional[str] = Query(None),
) -> AggregateRoundResponse:
    """
    Aggregate LoRA adapters for a round using FedAvg.
    
    Args:
        round_id: Round identifier
        request: Aggregation request
        
    Returns:
        AggregateRoundResponse with aggregation results
    """
    _require_operator(operator_key)
    # Get round config
    config = get_lora_round(round_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    
    if config.state == "CLOSED":
        raise HTTPException(
            status_code=400,
            detail=f"Round {round_id} is already closed"
        )
    
    # Get all submissions
    submissions = lora_round_manager.get_submissions(round_id)
    if not submissions:
        raise HTTPException(
            status_code=400,
            detail=f"No adapter submissions for round {round_id}"
        )
    
    # Aggregate adapters
    aggregated_adapter = aggregate_lora_adapters(
        submissions,
        weight_by_samples=request.weight_by_samples
    )
    
    if aggregated_adapter is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to aggregate adapters"
        )
    
    # Generate new adapter version
    if config.adapter_version:
        adapter_version = next_version(config.adapter_version)
    else:
        adapter_version = "v1"
    
    # Evaluate adapter
    previous_loss = None  # TODO: Load from previous adapter if exists
    eval_result = evaluate_adapter(
        round_id=round_id,
        adapter_version=adapter_version,
        aggregated_adapter=aggregated_adapter,
        previous_adapter_loss=previous_loss
    )
    
    # Save adapter
    adapter_data = {
        "version": adapter_version,
        "round_id": round_id,
        "base_model_id": config.base_model_id,
        "adapter_state_dict": aggregated_adapter,
        "num_clients": len(submissions),
        "evaluation": {
            "loss": eval_result.evaluation_loss,
            "passed": eval_result.passed
        },
        "created_at": time.time()
    }
    
    try:
        adapter_store.save_model(adapter_version, adapter_data)
    except Exception as e:
        logger.error(f"Failed to save adapter {adapter_version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save aggregated adapter: {e}"
        )
    
    # Close round
    close_lora_round(round_id)
    
    return AggregateRoundResponse(
        round_id=round_id,
        adapter_version=adapter_version,
        status="aggregated",
        num_adapters=len(submissions),
        evaluation_passed=eval_result.passed,
        evaluation_loss=eval_result.evaluation_loss
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
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    endpoints = {
        "health": "GET /health",
        "register_client": "POST /client/register",
        "get_task": "GET /task/{client_id}",
        "submit_update": "POST /update",
        "aggregate_round": "GET /aggregate/{round_id}",
        "get_round_status": "GET /status/{round_id}",
        "get_model": "GET /model/{version}",
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
        "submit_lora_adapter": "POST /rounds/{round_id}/submit",
        "aggregate_lora_round": "POST /rounds/{round_id}/aggregate"
    }
    
    return {
        "message": "Federated Learning Coordinator API",
        "version": "1.0.0",
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

