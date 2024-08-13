from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 
from huggingface_hub import hf_hub_download
import numpy as np

model = None
ratio = 16000/270

def snr_apply(batch, rank=None, audio_column_name="audio", batch_size=32):
    global model
    if model is None:
        model = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
    if rank is not None or torch.cuda.device_count() > 0:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        model.to(device)

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size = batch_size)
    if rank:
        pipeline.to(torch.device(device))
    
    device = pipeline._models["segmentation"].device

    if isinstance(batch[audio_column_name], list):  
        snr = []
        c50 = []
        vad_durations = []
        for sample in batch[audio_column_name]:
            res = pipeline({"sample_rate": sample["sampling_rate"],
                            "waveform": torch.tensor(sample["array"][None, :]).to(device).float()})
            
            mask = np.full(res["snr"].shape, False)
            for (segment, _) in res["annotation"].itertracks():
                start = int(segment.start * ratio)
                end = int(segment.end * ratio)
                mask[start:end] = True
            mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

            vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))
            
            snr.append(res["snr"][mask].mean())
            c50.append(res["c50"][mask].mean())
            vad_durations.append(vad_duration)
        
        # 16ms window
        batch["snr"] = snr
        batch["c50"] = c50
        batch["speech_duration"] = vad_durations
        
    else:
        res = pipeline({"sample_rate": batch[audio_column_name]["sampling_rate"],
                        "waveform": torch.tensor(batch[audio_column_name]["array"][None, :]).to(device).float()})
        
        mask = np.full(res["snr"].shape, False)
        for (segment, _) in res["annotation"].itertracks():
            start = int(segment.start * ratio)
            end = int(segment.end * ratio)
            mask[start:end] = True
        mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

        vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))     
        
        batch["snr"] = res["snr"][mask].mean()
        batch["c50"] = res["c50"][mask].mean()
        batch["speech_duration"] = vad_duration
        
    return batch