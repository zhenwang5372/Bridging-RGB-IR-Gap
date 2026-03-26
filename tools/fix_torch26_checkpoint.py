"""
Monkey patch for PyTorch 2.6 compatibility with MMEngine checkpoints.

PyTorch 2.6 changed the default value of `weights_only` in `torch.load` from 
False to True, which breaks MMEngine checkpoint loading (contains HistoryBuffer, 
optimizer state, etc.).

This patch ensures all torch.load calls use weights_only=False.
"""

import torch

# Save original torch.load
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    """
    Patched torch.load that sets weights_only=False by default.
    
    This allows loading MMEngine checkpoints that contain non-tensor objects
    like HistoryBuffer, optimizer states, etc.
    """
    # If weights_only is not specified, set it to False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    
    return _original_torch_load(*args, **kwargs)


# Apply the patch
torch.load = patched_torch_load

print("✅ Applied PyTorch 2.6 compatibility patch: torch.load now uses weights_only=False by default")

