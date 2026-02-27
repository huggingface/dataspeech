"""Minimal stub of brouhaha model classes for checkpoint deserialization.

The ylacombe/brouhaha-best checkpoint references brouhaha.models.CustomPyanNetModel.
PyTorch Lightning needs these classes importable to load the checkpoint.
This module provides just the model architecture definitions (no other brouhaha deps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet

SNR_MIN = -15
SNR_MAX = 80
C50_MIN = -10
C50_MAX = 60


class ParametricSigmoid(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        return (self.beta - self.alpha) * F.sigmoid(x) + self.alpha


class CustomClassifier(nn.Module):
    def __init__(self, in_features, out_features: int) -> None:
        super().__init__()
        self.linears = nn.ModuleDict({
            'vad': nn.Linear(in_features, out_features),
            'snr': nn.Linear(in_features, 1),
            'c50': nn.Linear(in_features, 1),
        })

    def forward(self, x: torch.Tensor):
        out = dict()
        for mode, linear in self.linears.items():
            _output = linear(x)
            out[mode] = _output
        return out


class CustomActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleDict({
            'vad': nn.Sigmoid(),
            'snr': ParametricSigmoid(SNR_MAX, SNR_MIN),
            'c50': ParametricSigmoid(C50_MAX, C50_MIN),
        })

    def forward(self, x: torch.Tensor):
        out = list()
        for mode, activation in self.activations.items():
            _output = activation(x[mode])
            out.append(_output)
        out = torch.stack(out)
        out = rearrange(out, "n b t o -> b t (n o)")
        return out


class RegressiveSegmentationModelMixin(Model):
    def build(self):
        nb_classif = len(set(self.specifications.classes) - set(['snr', 'c50']))
        self.classifier = CustomClassifier(32 * 2, nb_classif)
        self.activation = CustomActivation()


class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )
        nb_classif = len(set(self.specifications.classes) - set(['snr', 'c50']))
        self.classifier = CustomClassifier(in_features, nb_classif)
        self.activation = CustomActivation()
