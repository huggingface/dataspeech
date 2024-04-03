from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
from datasets import load_dataset
from multiprocess import set_start_method
import torch
import argparse
from datasets import Audio



demucs = pretrained.get_model('htdemucs')
source = demucs.sources

def wrap_audio(audio, sr):
    return {
        "array": audio.cpu().numpy(),
        "sampling_rate": sr
    }


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
        
        batch["vocals"] = [wrap_audio(s[-1,:,:length].mean(0), demucs.samplerate) for (s,length) in zip(stems, wavs_length)]
        batch["others"] = [wrap_audio(s[:-1, :,:length].sum(0).mean(0), demucs.samplerate) for (s,length) in zip(stems, wavs_length)]
        
    else:
        audio = torch.tensor(batch["audio"]["array"].squeeze(), device=device).to(torch.float32)
        sample_rate = batch["audio"]["sampling_rate"]
        audio = convert_audio(
                audio, sample_rate, demucs.samplerate, demucs.audio_channels)
        stems = apply_model(demucs, audio[None])
        
        batch["vocals"] = wrap_audio(stems[0,-1].mean(0), demucs.samplerate)
        batch["others"] = wrap_audio(stems[0, :-1].sum(0).mean(0), demucs.samplerate)

    return batch
    
if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/loading_methods#datasets.load_dataset.path")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--dump_folder_path", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be separated.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size. Speeds up operations on GPU.")
    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)    


    num_proc = torch.cuda.device_count() if torch.cuda.device_count() > 1 else None

    updated_dataset = dataset.map(
        filter_stems,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=num_proc,
    )
    
    updated_dataset = updated_dataset.cast_column("vocals", Audio())
    updated_dataset = updated_dataset.cast_column("others", Audio())
    
    if args.dump_folder_path:
        print("Saving to disk...")
        updated_dataset.save_to_disk(args.dump_folder_path)
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            updated_dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            updated_dataset.push_to_hub(args.repo_id)
    

