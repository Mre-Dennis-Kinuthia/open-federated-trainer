"""
Base Model Registry Module

Manages base LLM models for federated LoRA fine-tuning.
Base models are fixed and shared across all clients.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("model_registry")


@dataclass
class BaseModelConfig:
    """Configuration for a base LLM model."""
    model_name: str  # HuggingFace model identifier
    model_type: str  # e.g., "llama", "mistral", "phi"
    max_seq_length: int = 512
    trust_remote_code: bool = False


class BaseModelRegistry:
    """
    Registry for base LLM models used in federated LoRA fine-tuning.
    
    Clients pull base models from HuggingFace and train LoRA adapters.
    Base model weights are never updated - only adapters are trained.
    """
    
    # Predefined base models
    SUPPORTED_MODELS: Dict[str, BaseModelConfig] = {
        "llama-7b": BaseModelConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            model_type="llama",
            max_seq_length=512
        ),
        "mistral-7b": BaseModelConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            model_type="mistral",
            max_seq_length=512
        ),
        "phi-2": BaseModelConfig(
            model_name="microsoft/phi-2",
            model_type="phi",
            max_seq_length=512
        ),
        "tiny-llama": BaseModelConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            max_seq_length=512
        ),
    }
    
    def __init__(self):
        """Initialize the base model registry."""
        self.registered_models: Dict[str, BaseModelConfig] = {}
        # Register default models
        for model_id, config in self.SUPPORTED_MODELS.items():
            self.registered_models[model_id] = config
    
    def register_model(self, model_id: str, config: BaseModelConfig) -> bool:
        """
        Register a new base model.
        
        Args:
            model_id: Unique identifier for the model
            config: Base model configuration
            
        Returns:
            True if registered successfully, False if already exists
        """
        if model_id in self.registered_models:
            logger.warning(f"Model {model_id} already registered")
            return False
        
        self.registered_models[model_id] = config
        logger.info(f"Registered base model: {model_id} ({config.model_name})")
        return True
    
    def get_model_config(self, model_id: str) -> Optional[BaseModelConfig]:
        """
        Get configuration for a base model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            BaseModelConfig if found, None otherwise
        """
        return self.registered_models.get(model_id)
    
    def list_models(self) -> list[str]:
        """
        List all registered base models.
        
        Returns:
            List of model identifiers
        """
        return list(self.registered_models.keys())
    
    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model exists, False otherwise
        """
        return model_id in self.registered_models

