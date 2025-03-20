"""
Configuration module for OSPA Transformer experiments.

This module defines various model and experiment configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import json
import os
import torch


@dataclass
class OSPAConfig:
    """Configuration for OSPA Transformer model."""
    
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    
    # OSPA-specific settings
    enforce_mode: str = "regularize"  # "regularize", "strict", or "init"
    ortho_penalty_weight: float = 0.01
    
    # Additional architecture settings
    norm_first: bool = False  # Pre-norm or post-norm
    max_seq_length: int = 512
    use_decoder: bool = True  # For encoder-only or seq2seq models
    
    # Optimization settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_scheduler_type: str = "cosine"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.enforce_mode in ["regularize", "strict", "init"], \
            f"enforce_mode must be one of ['regularize', 'strict', 'init'], got {self.enforce_mode}"
        
        assert self.activation in ["relu", "gelu"], \
            f"activation must be one of ['relu', 'gelu'], got {self.activation}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OSPAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, config_path: str) -> None:
        """Save config to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> "OSPAConfig":
        """Load config from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different model sizes
OSPA_BASE_CONFIG = OSPAConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

OSPA_SMALL_CONFIG = OSPAConfig(
    d_model=256,
    nhead=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1
)

OSPA_LARGE_CONFIG = OSPAConfig(
    d_model=768,
    nhead=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    dim_feedforward=3072,
    dropout=0.1
)

# Task-specific configurations

@dataclass
class GlueTaskConfig:
    """Configuration for GLUE benchmark tasks."""
    
    task_name: str  # Name of the GLUE task (e.g., "sst2", "mnli", "qqp")
    num_labels: int  # Number of output classes
    max_seq_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    
    # Task-specific hyperparameters (optional)
    task_specific_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationTaskConfig:
    """Configuration for machine translation tasks."""
    
    src_lang: str  # Source language
    tgt_lang: str  # Target language
    max_seq_length: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 10
    beam_size: int = 5  # For beam search decoding
    
    # Task-specific hyperparameters (optional)
    task_specific_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRATaskConfig:
    """Configuration for Long Range Arena (LRA) benchmark tasks."""
    
    task_name: str  # Name of the LRA task (e.g., "text", "retrieval", "image")
    max_seq_length: int  # Maximum sequence length for this task
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 10
    
    # Task-specific hyperparameters (optional)
    task_specific_params: Dict[str, Any] = field(default_factory=dict)


# Experiment configurations

@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    
    # Experiment metadata
    name: str
    description: str = ""
    output_dir: str = "./experiments"
    
    # Model configuration
    model_config: OSPAConfig = field(default_factory=lambda: OSPA_BASE_CONFIG)
    
    # Task configuration
    task_config: Union[GlueTaskConfig, TranslationTaskConfig, LRATaskConfig] = None
    
    # Training configuration
    seed: int = 42
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000
    
    # Hardware configuration
    use_cuda: bool = True
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 4  # For data loading
    
    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_device(self) -> torch.device:
        """Get device for training."""
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def save(self, config_path: str = None) -> str:
        """Save experiment configuration to JSON file."""
        if config_path is None:
            config_path = os.path.join(self.output_dir, f"{self.name}_config.json")
        
        config_dict = {
            "name": self.name,
            "description": self.description,
            "output_dir": self.output_dir,
            "model_config": self.model_config.to_dict(),
            "task_config": self.task_config.__dict__ if self.task_config else None,
            "seed": self.seed,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "use_cuda": self.use_cuda,
            "use_amp": self.use_amp,
            "num_workers": self.num_workers
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path


# Predefined experiment configurations for different tasks

def get_sst2_experiment(enforce_mode="regularize", ortho_penalty_weight=0.01):
    """Get SST-2 sentiment analysis experiment configuration."""
    model_config = OSPAConfig(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=0,
        use_decoder=False,
        enforce_mode=enforce_mode,
        ortho_penalty_weight=ortho_penalty_weight
    )
    
    task_config = GlueTaskConfig(
        task_name="sst2",
        num_labels=2,
        max_seq_length=128,
        batch_size=32,
        num_epochs=3
    )
    
    return ExperimentConfig(
        name=f"sst2_{enforce_mode}",
        description=f"SST-2 sentiment analysis using OSPA with {enforce_mode} orthogonality",
        model_config=model_config,
        task_config=task_config
    )


def get_iwslt_experiment(enforce_mode="regularize", ortho_penalty_weight=0.01):
    """Get IWSLT translation experiment configuration."""
    model_config = OSPAConfig(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        use_decoder=True,
        enforce_mode=enforce_mode,
        ortho_penalty_weight=ortho_penalty_weight,
        max_seq_length=256
    )
    
    task_config = TranslationTaskConfig(
        src_lang="de",
        tgt_lang="en",
        max_seq_length=256,
        batch_size=16,
        num_epochs=15
    )
    
    return ExperimentConfig(
        name=f"iwslt_de_en_{enforce_mode}",
        description=f"IWSLT German-English translation using OSPA with {enforce_mode} orthogonality",
        model_config=model_config,
        task_config=task_config
    )


def get_lra_text_experiment(enforce_mode="regularize", ortho_penalty_weight=0.01):
    """Get LRA text classification experiment configuration."""
    model_config = OSPAConfig(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=0,
        use_decoder=False,
        enforce_mode=enforce_mode,
        ortho_penalty_weight=ortho_penalty_weight,
        max_seq_length=4096  # Long sequences for LRA
    )
    
    task_config = LRATaskConfig(
        task_name="text",
        max_seq_length=4096,
        batch_size=8,  # Smaller batch size for longer sequences
        num_epochs=10
    )
    
    return ExperimentConfig(
        name=f"lra_text_{enforce_mode}",
        description=f"LRA text classification using OSPA with {enforce_mode} orthogonality",
        model_config=model_config,
        task_config=task_config
    )