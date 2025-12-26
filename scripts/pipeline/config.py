"""
Pipeline configuration management
"""
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class PipelineConfig:
    """Pipeline configuration"""
    
    DEFAULT_CONFIG = {
        "training": {
            "lora": {"enabled": True},
            "fusion": {"enabled": True}
        },
        "kb_building": {"enabled": True},
        "evaluation": {
            "retrieval": {"enabled": True},
            "encoders": {"enabled": True},
            "counterfactual": {"enabled": True},
            "lora": {"enabled": True}
        }
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = Path("outputs/experiments") / self.timestamp
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self._merge_config(config_dict)
    
    def _merge_config(self, custom: Dict):
        """Deep merge custom config"""
        def merge(base, custom):
            for key, value in custom.items():
                if isinstance(value, dict) and key in base:
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(self.config, custom)
    
    def save(self):
        """Save config to experiment directory"""
        config_path = self.output_root / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        return config_path
    
    def get(self, key_path: str, default=None):
        """Get nested config value"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from file or defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                custom_config = json.load(f)
            else:
                custom_config = yaml.safe_load(f)
        return PipelineConfig(custom_config)
    return PipelineConfig()
