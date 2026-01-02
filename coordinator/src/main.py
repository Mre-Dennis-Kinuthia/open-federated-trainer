"""
FastAPI Server for Federated Learning Coordinator

Main entry point for the coordinator API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from core.round_manager import RoundManager
from core.task_assigner import TaskAssigner
from core.update_validator import UpdateValidator
from core.aggregator import Aggregator
from core.model_store import ModelStore
from core.metrics import MetricsCollector
from core.auth import AuthManager
from core.rate_limiter import RateLimiter
from core.privacy import PrivacyProtector
from core.async_round_manager import AsyncRoundManager, AsyncRoundConfig
from core.reputation import ReputationManager
from core.incentives import IncentiveManager
from utils.logger import setup_coordinator_logger
import os

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

# Initialize core modules
model_store = ModelStore()
round_manager = RoundManager()
task_assigner = TaskAssigner(round_manager, model_store)
auth_manager = AuthManager()
rate_limiter = RateLimiter()
privacy_protector = PrivacyProtector()
update_validator = UpdateValidator(
    round_manager,
    auth_manager=auth_manager,
    rate_limiter=rate_limiter,
    privacy_protector=privacy_protector
)
metrics_collector = MetricsCollector()
aggregator = Aggregator(round_manager, model_store, task_assigner, metrics_collector, rate_limiter)

# Initialize advanced modules (optional)
enable_async = os.getenv("ENABLE_ASYNC_ROUNDS", "false").lower() == "true"
async_config = AsyncRoundConfig(
    minimum_updates_required=int(os.getenv("ASYNC_MIN_UPDATES", "2")),
    max_round_duration_seconds=float(os.getenv("ASYNC_MAX_DURATION", "300.0")),
    enable_async=enable_async
)
async_round_manager = AsyncRoundManager(round_manager, async_config) if enable_async else None
reputation_manager = ReputationManager()
incentive_manager = IncentiveManager(
    base_reward_per_update=float(os.getenv("INCENTIVE_BASE_REWARD", "10.0")),
    speed_bonus_threshold=float(os.getenv("INCENTIVE_SPEED_THRESHOLD", "30.0")),
    consistency_bonus_threshold=int(os.getenv("INCENTIVE_CONSISTENCY_THRESHOLD", "5"))
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


@app.post("/client/register", response_model=ClientRegisterResponse)
async def register_client(request: ClientRegisterRequest) -> ClientRegisterResponse:
    """
    Register a new client and receive an API key.
    
    Args:
        request: Client registration request with client_name
        
    Returns:
        Registration response with success status and API key
    """
    logger.info(f"Registration request received for client {request.client_name}", extra={
        "component": "coordinator",
        "event": "registration_request",
        "client_id": request.client_name
    })
    
    # Register with round manager
    success = round_manager.register_client(request.client_name)
    
    if success:
        # Generate and register API key
        try:
            api_key = auth_manager.register_client(request.client_name)
            
            # Record in metrics
            metrics_collector.total_clients_seen.add(request.client_name)
            
            logger.info(f"Client {request.client_name} registered with API key", extra={
                "component": "coordinator",
                "event": "client_registered",
                "client_id": request.client_name
            })
            
            return ClientRegisterResponse(
                success=True,
                message=f"Client {request.client_name} registered successfully. Save your API key!",
                client_id=request.client_name,
                api_key=api_key
            )
        except ValueError as e:
            # Client already has API key (shouldn't happen, but handle gracefully)
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Client {request.client_name} is already registered"
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
async def aggregate_round(round_id: int) -> AggregateResponse:
    """
    Aggregate all updates for a round.
    
    Args:
        round_id: Identifier of the round to aggregate
        
    Returns:
        Aggregation result with aggregated model information
    """
    result = aggregator.aggregate(round_id)
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_id} not found"
        )
    
    return AggregateResponse(
        round_id=result["round_id"],
        model_version=result["model_version"],
        status=result["status"],
        aggregated_model=result["aggregated_model"],
        num_updates=result["num_updates"]
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    endpoints = {
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
        "get_async_round_stats": "GET /async/round/{round_id}/stats"
    }
    
    return {
        "message": "Federated Learning Coordinator API",
        "version": "1.0.0",
        "async_enabled": enable_async,
        "endpoints": endpoints
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

