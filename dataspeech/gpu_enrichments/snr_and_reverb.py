from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from pathlib import Path

import warnings
with warnings.catch_warnings():
    # Newer pyannote.audio (3.3+) warns about torchcodec at import time.
    # torchcodec is not needed here — we pass pre-loaded waveforms, not files.
    warnings.filterwarnings("ignore", message="torchcodec", category=UserWarning)
    from pyannote.audio import Model, Inference
    from pyannote.audio.core.model import Specifications
    from pyannote.audio.core.task import Resolution
    from pyannote.audio.utils.multi_task import map_with_specifications
    from pyannote.audio.utils.signal import Binarize
    from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature


class _BrouhahaInference(Inference):
    """Inference subclass that uses repeat-based padding for the last chunk,
    matching the original brouhaha-vad BrouhahaInference behavior exactly."""

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        hook: Optional[Callable],
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:

        window_size: int = self.model.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        def __frames(
            receptive_field, specifications: Optional[Specifications] = None
        ) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return receptive_field

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.model.specifications, __frames, self.model.receptive_field
        )

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0

        if has_last_chunk:
            # repeat last chunk to fill window (brouhaha-style, not zero-pad)
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size:]
            channel, last_window_size = last_chunk.shape
            num_repeat = window_size // last_window_size + 1
            last_chunk = last_chunk.repeat((channel, num_repeat))
            last_chunk = last_chunk[:, :window_size]

        def __empty_list(**kwargs):
            return list()

        outputs: Union[
            List[np.ndarray], Tuple[List[np.ndarray]]
        ] = map_with_specifications(self.model.specifications, __empty_list)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs) -> None:
            output.append(batch_output)
            return

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]

            batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch)

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, batch_outputs
            )

            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        # process orphan last chunk
        if has_last_chunk:
            last_outputs = self.infer(last_chunk[None])

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, last_outputs
            )

            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )

        def __vstack(output: List[np.ndarray], **kwargs) -> np.ndarray:
            return np.vstack(output)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = map_with_specifications(
            self.model.specifications, __vstack, outputs
        )

        def __aggregate(
            outputs: np.ndarray,
            frames: SlidingWindow,
            specifications: Optional[Specifications] = None,
        ) -> SlidingWindowFeature:
            if (
                self.skip_aggregation
                or specifications.resolution == Resolution.CHUNK
                or (
                    specifications.permutation_invariant
                    and self.pre_aggregation_hook is None
                )
            ):
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                return SlidingWindowFeature(outputs, frames)

            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)

            aggregated = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(start=0.0, duration=self.duration, step=self.step),
                ),
                frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
            )

            # remove padding that was added to last chunk
            if has_last_chunk:
                aggregated.data = aggregated.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )

            return aggregated

        return map_with_specifications(
            self.model.specifications, __aggregate, outputs, frames
        )


model = None
ratio = 16000/270

# Binarize with brouhaha's default parameters (onset=offset=0.780, no min duration)
_binarize = Binarize(onset=0.780, offset=0.780)


def _run_inference(sample, inference):
    """Run inference on a single audio sample and return annotation, snr, c50."""
    segmentations = inference({"sample_rate": sample["sampling_rate"],
                               "waveform": torch.tensor(sample["array"][None, :]).to(inference.device).float()})

    # Extract VAD column and binarize (replicates RegressiveActivityDetectionPipeline.apply)
    vad_scores = SlidingWindowFeature(
        np.expand_dims(segmentations.data[:, 0], axis=1),
        segmentations.sliding_window,
    )
    annotation = _binarize(vad_scores)

    snr_array = segmentations.data[:, 1]
    c50_array = segmentations.data[:, 2]

    return annotation, snr_array, c50_array


def snr_apply(batch, rank=None, audio_column_name="audio", batch_size=32):
    global model
    if model is None:
        import sys
        from dataspeech.gpu_enrichments import _brouhaha_compat
        import types

        # Register stub so that the checkpoint's reference to brouhaha.models
        # resolves without installing the full brouhaha-vad package.
        if "brouhaha" not in sys.modules:
            brouhaha_pkg = types.ModuleType("brouhaha")
            brouhaha_pkg.__path__ = []
            sys.modules["brouhaha"] = brouhaha_pkg
        if "brouhaha.models" not in sys.modules:
            sys.modules["brouhaha.models"] = _brouhaha_compat

        # The ylacombe/brouhaha-best checkpoint contains pickle-serialized objects
        # (e.g. TorchVersion) that fail with torch.load's weights_only=True default
        # in PyTorch 2.6+. Temporarily patch torch.load during model loading.
        _original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_load
        try:
            model = Model.from_pretrained(
                Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
                strict=False,
            )
        finally:
            torch.load = _original_torch_load
    if rank is not None or torch.cuda.device_count() > 0:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        model.to(device)

    inference = _BrouhahaInference(model, device=torch.device(model.device))

    if isinstance(batch[audio_column_name], list):
        snr = []
        c50 = []
        vad_durations = []
        for sample in batch[audio_column_name]:
            annotation, snr_array, c50_array = _run_inference(sample, inference)

            mask = np.full(snr_array.shape, False)
            for (segment, _) in annotation.itertracks():
                start = int(segment.start * ratio)
                end = int(segment.end * ratio)
                mask[start:end] = True
            mask =  (~((snr_array == 0.0) & (c50_array == 0.0)) & mask)

            vad_duration = sum(map(lambda x: x[0].duration, annotation.itertracks()))

            snr.append(snr_array[mask].mean())
            c50.append(c50_array[mask].mean())
            vad_durations.append(np.float32(vad_duration))

        # 16ms window
        batch["snr"] = snr
        batch["c50"] = c50
        batch["speech_duration"] = vad_durations

    else:
        annotation, snr_array, c50_array = _run_inference(batch[audio_column_name], inference)

        mask = np.full(snr_array.shape, False)
        for (segment, _) in annotation.itertracks():
            start = int(segment.start * ratio)
            end = int(segment.end * ratio)
            mask[start:end] = True
        mask =  (~((snr_array == 0.0) & (c50_array == 0.0)) & mask)

        vad_duration = sum(map(lambda x: x[0].duration, annotation.itertracks()))

        batch["snr"] = snr_array[mask].mean()
        batch["c50"] = c50_array[mask].mean()
        batch["speech_duration"] = vad_duration

    return batch
