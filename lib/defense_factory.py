# defense_factory.py
from lib.defenses import SmoothLLM, Empty

DEFENSE_REGISTRY = {
    "smoothllm": SmoothLLM,
    "empty": Empty
}

def get_defense(defense_type: str, *args, **kwargs):
    cls = DEFENSE_REGISTRY.get(defense_type.lower())
    if cls is None:
        raise ValueError(f"Unknown defense: '{defense_type}'. Choose from: {list(DEFENSE_REGISTRY)}")
    return cls(*args, **kwargs)