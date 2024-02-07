from datasets import load_dataset
from multiprocess import set_start_method
import torch 

import penn


# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065


# TODO: make compatible with streaming
# TODO: make compatible with other naming
def pitch_apply(batch, rank=None):
    if isinstance(batch["audio"], list):  
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for sample in batch["audio"]:
            # Infer pitch and periodicity
            pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=rank)
            
            utterance_pitch_mean.append(pitch.mean())
            utterance_pitch_std.append(pitch.std())
            
        batch["utterance_pitch_mean"] = utterance_pitch_mean 
        batch["utterance_pitch_std"] = utterance_pitch_std 
    else:
        pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=rank)
        
        batch["utterance_pitch_mean"] = pitch.mean()
        batch["utterance_pitch_std"] = pitch.std()

    return batch
    
if __name__ == "__main__":
    set_start_method("spawn")
    
    # sampling_rate = next(iter(dataset))["audio"]["sampling_rate"]    
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    dataset = load_dataset("ylacombe/english_dialects", "irish_male")

    updated_dataset = dataset.map(
        pitch_apply,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count()
    )
    
    updated_dataset.save_to_disk("artefacts/english_dialects_irish")
    # updated_dataset.push_to_hub("ylacombe/english_dialects", "irish_male")
    
    print("ok")
    
