from datasets import load_dataset
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply
import torch
import argparse
from accelerate.logging import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Repo id or local path.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--dump_folder_path", default=None, type=str, help="If specified, save the dasaset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Audio column name.")
    parser.add_argument("--text_column_name", default="text", type=str, help="Text column name.")
    parser.add_argument("--rename_column", action="store_true", help="If activated, rename audio and text column names to 'audio' and 'text'. Useful if you wan't to merge datasets afterwards.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")

    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)
        
    audio_column_name = "audio" if args.rename_column else args.audio_column_name
    text_column_name = "text" if args.rename_column else args.text_column_name
    if args.rename_column:
        dataset = dataset.rename_columns({args.audio_column_name: "audio", args.text_column_name: "text"})


    logger.info("Compute speaking rate")
    updated_dataset = dataset.map(
        rate_apply,
        with_rank=False,
        num_proc=args.cpu_num_workers,
    )

    logger.info("Compute snr and reverb")
    updated_dataset = updated_dataset.map(
        snr_apply,
        batched=True,
        batch_size=16,
        with_rank=True if torch.cuda.device_count()>0 else False,
        num_proc=torch.cuda.device_count() if torch.cuda.device_count()>0 else args.cpu_num_workers
    )
    
    logger.info("Compute pitch")
    updated_dataset = updated_dataset.map(
        pitch_apply,
        batched=True,
        batch_size=16,
        with_rank=True if torch.cuda.device_count()>0 else False,
        num_proc=torch.cuda.device_count() if torch.cuda.device_count()>0 else args.cpu_num_workers
    )
    
    updated_dataset = updated_dataset.remove_columns(audio_column_name)
    
    if args.dump_folder_path:
        logger.info("Saving to disk...")
        updated_dataset.save_to_disk(args.dump_folder_path)
    if args.repo_id:
        logger.info("Pushing to the hub...")
        if args.configuration:
            updated_dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            updated_dataset.push_to_hub(args.repo_id)
    
