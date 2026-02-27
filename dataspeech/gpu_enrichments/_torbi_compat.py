"""Compatibility shim for torbi on unsupported torch/CUDA versions.

torbi ships prebuilt C++ binaries for specific torch+CUDA combinations.
When no matching binary exists (e.g. torch 2.9+), import fails with
FileNotFoundError. This module patches torbi to fall back to a pure-Python
Viterbi implementation using librosa.sequence.viterbi — the same algorithm
used by torbi's own reference implementation (torbi.reference.core).
"""

import sys


def ensure_torbi():
    """Ensure torbi is importable, patching with a librosa fallback if needed."""
    if 'torbi' in sys.modules:
        return

    try:
        import torbi  # noqa: F401
        return
    except FileNotFoundError:
        pass

    # torbi's C++ binary is unavailable for this torch/CUDA version.
    # Create a minimal stub module providing only from_probabilities(),
    # which is the sole function penn uses from torbi.
    import types

    import numpy as np
    import torch

    torbi_mod = types.ModuleType('torbi')
    torbi_mod.__package__ = 'torbi'
    sys.modules['torbi'] = torbi_mod

    def from_probabilities(
        observation,
        batch_frames=None,
        transition=None,
        initial=None,
        log_probs=False,
        gpu=None,
        num_threads=1,
    ):
        """Pure-Python Viterbi decoding via librosa (fallback for missing C++ binary).

        Replicates the interface of torbi.core.from_probabilities but uses
        librosa.sequence.viterbi for the actual decoding, matching the
        algorithm in torbi.reference.core.from_probabilities.
        """
        import librosa

        device = observation.device
        batch, frames, states = observation.shape

        # Convert to probability space for librosa
        obs_probs = torch.exp(observation) if log_probs else observation
        obs_np = obs_probs.to(torch.float32).cpu().numpy()

        # Initial distribution (librosa expects probabilities, not log-probs)
        if initial is None:
            initial_np = np.full((states,), 1.0 / states, dtype=np.float32)
        else:
            init_t = torch.exp(initial) if log_probs else initial
            initial_np = init_t.to(torch.float32).cpu().numpy()

        # Transition matrix (librosa expects probabilities)
        if transition is None:
            trans_np = np.full(
                (states, states), 1.0 / states, dtype=np.float32)
        else:
            trans_t = torch.exp(transition) if log_probs else transition
            trans_np = trans_t.to(torch.float32).cpu().numpy()

        # Decode each batch item
        results = []
        for i in range(batch):
            n = batch_frames[i].item() if batch_frames is not None else frames
            indices = librosa.sequence.viterbi(
                obs_np[i, :n].T, trans_np, p_init=initial_np)
            if n < frames:
                padded = np.zeros(frames, dtype=np.int32)
                padded[:n] = indices
                indices = padded
            results.append(torch.tensor(
                indices.astype(np.int32), dtype=torch.int, device=device))

        return torch.stack(results)

    torbi_mod.from_probabilities = from_probabilities
