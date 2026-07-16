"""
LoRA Trainer Module

Trains LoRA adapters on local data using PyTorch and PEFT.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import os

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from peft import prepare_model_for_kbit_training
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/peft not available. LoRA training disabled.")


from .dataset_loader import load_local_dataset, DatasetConfig
from .metrics import TrainingMetrics
from utils.logger import get_logger

logger = get_logger("lora_trainer")


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning of language models.
    
    Loads base model in 4-bit mode, initializes LoRA adapters,
    and trains only adapter parameters.
    """
    
    def __init__(
        self,
        base_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list[str]] = None,
        max_seq_length: int = 512,
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            base_model_name: HuggingFace model identifier
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of modules to apply LoRA to
            max_seq_length: Maximum sequence length
            use_4bit: Use 4-bit quantization (requires bitsandbytes)
            device_map: Device mapping strategy
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft are required for LoRA training")
        
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_model(self, previous_adapter_state: Optional[Dict] = None):
        """
        Load base model and initialize LoRA adapters.
        
        Args:
            previous_adapter_state: Previous aggregated adapter weights
        """
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=os.getenv("TRUST_REMOTE_MODEL_CODE", "false").lower()
            in {"1", "true", "yes"}
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if self.use_4bit:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    load_in_4bit=True,
                    device_map=self.device_map,
                    trust_remote_code=False,
                    torch_dtype=torch.float16
                )
                logger.info("Loaded model in 4-bit mode")
            except Exception as e:
                logger.warning(f"4-bit loading failed: {e}. Falling back to CPU.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map="cpu",
                    trust_remote_code=False,
                    torch_dtype=torch.float32
                )
                self.use_4bit = False
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device_map,
                trust_remote_code=False,
                torch_dtype=torch.float32
            )
        
        # Prepare for k-bit training if using 4-bit
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        if previous_adapter_state:
            state = {
                key: torch.tensor(value)
                for key, value in previous_adapter_state.items()
            }
            load_result = set_peft_model_state_dict(self.peft_model, state)
            unexpected = getattr(load_result, "unexpected_keys", [])
            if unexpected:
                raise ValueError(
                    f"Previous adapter has unexpected keys: {unexpected[:5]}"
                )
            logger.info("Loaded previous aggregated adapter weights")
        else:
            logger.info("Initialized new LoRA adapters")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable()
    
    def train(
        self,
        texts: list[str],
        max_steps: int = 100,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 10
    ) -> Tuple[Dict, TrainingMetrics]:
        """
        Train LoRA adapters on local data.
        
        Args:
            texts: List of training texts
            max_steps: Maximum training steps
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            
        Returns:
            Tuple of (adapter_state_dict, training_metrics)
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Starting LoRA training: {len(texts)} samples, {max_steps} steps")
        
        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
        
        tokenized = tokenize_function(texts)
        
        # Create dataset
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        dataset = TextDataset(tokenized)
        if len(dataset) < 2:
            raise ValueError("LoRA training requires at least two text samples")
        eval_size = max(1, min(len(dataset) // 10, 64))
        train_size = len(dataset) - eval_size
        generator = torch.Generator().manual_seed(
            int(os.getenv("LORA_DATASET_SEED", "42"))
        )
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, eval_size],
            generator=generator,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./lora_output",
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints
            fp16=self.use_4bit,
            bf16=False,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Get initial loss
        initial_loss = float(trainer.evaluate()["eval_loss"])
        
        # Train
        train_result = trainer.train()
        
        # Get final loss
        final_loss = float(trainer.evaluate()["eval_loss"])
        
        # Extract adapter weights
        adapter_state_dict = get_peft_model_state_dict(self.peft_model)
        
        # Convert tensors to lists for JSON serialization
        serialized_adapter = {}
        for key, value in adapter_state_dict.items():
            serialized_adapter[key] = value.cpu().numpy().tolist()
        
        # Create metrics
        metrics = TrainingMetrics(
            num_samples=len(texts),
            num_steps=max_steps,
            initial_loss=initial_loss,
            final_loss=final_loss,
            learning_rate=learning_rate
        )
        
        logger.info(f"Training completed: final_loss={final_loss:.4f}")
        
        return serialized_adapter, metrics


def train_lora_adapter(
    base_model_name: str,
    texts: list[str],
    round_config: Dict,
    previous_adapter_state: Optional[Dict] = None
) -> Tuple[Dict, TrainingMetrics]:
    """
    Train a LoRA adapter for federated learning.
    
    This is the main entry point for client-side LoRA training.
    
    Args:
        base_model_name: HuggingFace model identifier
        texts: List of training texts
        round_config: Round configuration from coordinator
        previous_adapter_state: Previous aggregated adapter weights (optional)
        
    Returns:
        Tuple of (adapter_state_dict, training_metrics)
    """
    trainer = LoRATrainer(
        base_model_name=base_model_name,
        lora_r=round_config.get("lora_r", 8),
        lora_alpha=round_config.get("lora_alpha", 16),
        lora_dropout=round_config.get("lora_dropout", 0.1),
        target_modules=round_config.get("target_modules", ["q_proj", "v_proj"]),
        max_seq_length=round_config.get("max_seq_length", 512)
    )
    
    trainer.load_model(previous_adapter_state)
    
    adapter, metrics = trainer.train(
        texts=texts,
        max_steps=round_config.get("max_steps", 100),
        learning_rate=round_config.get("learning_rate", 2e-4),
        batch_size=round_config.get("batch_size", 4),
        gradient_accumulation_steps=round_config.get("gradient_accumulation_steps", 4),
        warmup_steps=round_config.get("warmup_steps", 10)
    )
    
    return adapter, metrics

