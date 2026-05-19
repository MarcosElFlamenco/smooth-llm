# defense_factory.py
from lib.defenses import SmoothLLM, Empty

DEFENSE_REGISTRY = {
    "smoothllm": SmoothLLM,
    "empty": Empty
}


def get_defense(defense_type: str, target_model, args=None):
    cls = DEFENSE_REGISTRY.get(defense_type.lower())
    if cls is None:
        raise ValueError(f"Unknown defense: '{defense_type}'. Choose from: {list(DEFENSE_REGISTRY)}")
    if cls == Empty:
        return Empty(target_model)
    elif cls == SmoothLLM:
        return SmoothLLM(
            target_model=target_model,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies
        )