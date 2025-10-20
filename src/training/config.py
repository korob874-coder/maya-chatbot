# src/training/config.py
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_length: int = 256
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_dir: str = "./checkpoints"
    
@dataclass
class DataConfig:
    topics: List[str] = None
    language: str = "id"
    min_text_length: int = 1000
    cache_dir: str = "./data/cache"
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = [
                "Kecerdasan buatan", "Pembelajaran mesin", 
                "Python", "Ilmu komputer", "Data science"
            ]

class MayaConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_to_yaml(self, path: str):
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
