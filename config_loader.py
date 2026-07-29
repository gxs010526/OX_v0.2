# config_loader.py
import json
from typing import Dict, Any

def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf8') as f:
        cfg = json.load(f)
    return cfg
