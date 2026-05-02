"""
Synthetic expansion hooks (seed logs → controlled mutations).

Intended for future augmentation without coupling to a specific generator model.
"""

from logs_to_training.expand.synthetic_hooks import SyntheticExpansionConfig, apply_synthetic_mutations

__all__ = ["SyntheticExpansionConfig", "apply_synthetic_mutations"]
