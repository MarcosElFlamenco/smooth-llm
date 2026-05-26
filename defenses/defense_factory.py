# defense_factory.py
from defenses.defenses import NoDefense

DEFENSE_REGISTRY = {
    "nodefense": NoDefense 
}


def get_defense(defense_type: str, target_model, tokenizer, conv_template, args=None):
    cls = DEFENSE_REGISTRY.get(defense_type.lower())
    if cls is None:
        raise ValueError(f"Unknown defense: '{defense_type}'. Choose from: {list(DEFENSE_REGISTRY)}")
    if cls == NoDefense:
        return NoDefense(
            target_model=target_model,
            tokenizer=tokenizer,
            conv_template=conv_template
        )
        