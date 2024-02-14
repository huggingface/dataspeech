from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
from datasets import load_dataset
from multiprocess import set_start_method
import torch


demucs = pretrained.get_model('htdemucs')
source = demucs.sources

def wrap_audio(audio, sr):
    return {
        "array": audio.cpu().numpy().T,
        "sampling_rate": sr
    }


# TODO: make compatible with streaming
# TODO: make compatible with other naming and stems
def filter_stems(batch, rank=None):
    if rank is not None:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        demucs.to(device)

    if isinstance(batch["audio"], list):  
        wavs = [convert_audio(
                    torch.tensor(audio["array"][None], device=device).to(torch.float32), audio["sampling_rate"], demucs.samplerate, demucs.audio_channels).T for audio in batch["audio"]]
        wavs_length = [audio.shape[0] for audio in wavs]
        
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0).transpose(1,2)
        stems = apply_model(demucs, wavs)
        



        batch["vocals"] = [wrap_audio(s[-1,:,:length], demucs.samplerate) for (s,length) in zip(stems, wavs_length)]
        batch["others"] = [wrap_audio(s[:-1, :,:length].sum(0), demucs.samplerate) for (s,length) in zip(stems, wavs_length)]
        
    else:
        audio = torch.tensor(batch["audio"]["array"].squeeze(), device=device).to(torch.float32)
        sample_rate = batch["audio"]["sampling_rate"]
        audio = convert_audio(
                audio, sample_rate, demucs.samplerate, demucs.audio_channels)
        stems = apply_model(demucs, audio[None])
        
        batch["vocals"] = wrap_audio(stems[0,-1], demucs.samplerate)
        batch["others"] = wrap_audio(stems[0, :-1].sum(0), demucs.samplerate)

    return batch
    
if __name__ == "__main__":
    set_start_method("spawn")
    dataset = load_dataset("patrickvonplaten/bella_ciao")
    
    # sampling_rate = next(iter(dataset))["audio"]["sampling_rate"]    
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    updated_dataset = dataset.map(
        filter_stems,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count()
    )
    
    from datasets import Audio
    updated_dataset = updated_dataset.cast_column("vocals", Audio())
    updated_dataset = updated_dataset.cast_column("others", Audio())
    
    updated_dataset.save_to_disk("artefacts/english_dialects_irish")
    # updated_dataset.push_to_hub("patrickvonplaten/bella_ciao")
    
    print("ok")
    
