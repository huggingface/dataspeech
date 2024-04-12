import torch 
import penn


# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065


def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    if isinstance(batch[audio_column_name], list):  
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for sample in batch[audio_column_name]:
            # Infer pitch and periodicity
            pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if rank else rank
                )
            
            utterance_pitch_mean.append(pitch.mean().cpu())
            utterance_pitch_std.append(pitch.std().cpu())
            
        batch[f"{output_column_name}_mean"] = utterance_pitch_mean 
        batch[f"{output_column_name}_std"] = utterance_pitch_std 
    else:
        pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if rank else rank
                )        
        batch[f"{output_column_name}_mean"] = pitch.mean().cpu()
        batch[f"{output_column_name}_std"] = pitch.std().cpu()

    return batch