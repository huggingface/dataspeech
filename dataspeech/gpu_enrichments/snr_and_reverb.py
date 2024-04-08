from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 
from huggingface_hub import hf_hub_download

model = None

def snr_apply(batch, rank=None, audio_column_name="audio"):
    global model
    if model is None:
        model = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
    if rank is not None:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        model.to(device)

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
    if rank:
        pipeline.to(torch.device(device))
    
    device = pipeline._models["segmentation"].device

    if isinstance(batch[audio_column_name], list):  
        snr = []
        c50 = []
        for sample in batch[audio_column_name]:
            res = pipeline({"sample_rate": sample["sampling_rate"],
                            "waveform": torch.tensor(sample["array"][None, :]).to(device).float()})
            
            snr.append(res["snr"].mean())
            c50.append(res["c50"].mean())
        
        batch["snr"] = snr
        batch["c50"] = c50
    else:
        res = pipeline({"sample_rate": batch[audio_column_name]["sampling_rate"],
                        "waveform": torch.tensor(batch[audio_column_name]["array"][None, :]).to(device).float()})
        
        batch["snr"] = res["snr"].mean()
        batch["c50"] = res["c50"].mean()

    return batch