from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from datasets import load_dataset
from multiprocess import set_start_method
import torch 


model = Model.from_pretrained(
    Path("artefacts/best.ckp"),
    strict=False,
)

# TODO: make compatible with streaming
# TODO: make compatible with other naming
def vad_apply(batch, rank=None):
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
    
if __name__ == "__main__":
    set_start_method("spawn")
    dataset = load_dataset("ylacombe/english_dialects", "irish_male")
    
    # sampling_rate = next(iter(dataset))["audio"]["sampling_rate"]    
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    updated_dataset = dataset.map(
        vad_apply,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count()
    )
    
    updated_dataset.save_to_disk("artefacts/english_dialects_irish")
    # updated_dataset.push_to_hub("ylacombe/english_dialects", "irish_male")
    
    print("ok")
    
