RUNTIME_MODES = {
    "deterministic": {
        "signals": ["probability"],
        "effective_forward_passes": 1,
    },
    "fast": {
        "signals": ["probability", "entropy", "margin", "component_stats"],
        "effective_forward_passes": 1,
    },
    "balanced": {
        "signals": ["probability", "entropy", "margin", "component_stats", "tta"],
        "effective_forward_passes": 6,
    },
    "full": {
        "signals": ["probability", "entropy", "margin", "component_stats", "tta", "mc_dropout", "shift"],
        "effective_forward_passes": "20+6+4",
    },
}
