from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 

model = None

def snr_apply(batch, rank=None):
    global model
    if model is None:
        model = Model.from_pretrained(
            Path("artefacts/best.ckp"),
            strict=False,
        )
    if rank:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        model.to(device)

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
    if rank:
        pipeline.to(torch.device(device))
    
    device = pipeline._models["segmentation"].device

    if isinstance(batch["audio"], list):  
        snr = []
        c50 = []
        for sample in batch["audio"]:
            res = pipeline({"sample_rate": sample["sampling_rate"],
                            "waveform": torch.tensor(sample["array"][None, :]).to(device).float()})
            
            snr.append(res["snr"].mean())
            c50.append(res["c50"].mean())
        
        batch["snr"] = snr
        batch["c50"] = c50
    else:
        res = pipeline({"sample_rate": batch["audio"]["sampling_rate"],
                        "waveform": torch.tensor(batch["audio"]["array"][None, :]).to(device).float()})
        
        batch["snr"] = res["snr"].mean()
        batch["c50"] = res["c50"].mean()

    return batch