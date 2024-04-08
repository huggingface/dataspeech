from datasets import load_dataset
from multiprocess import set_start_method
import pandas as pd
import argparse


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Repo id or local path.")
    parser.add_argument("tsv_path", default=None, type=str, help="Text column name.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dasaset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--speaker_id_column_name", default="speaker_id", type=str, help="Audio column name.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")

    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)
        
    speaker_id_column_name = args.speaker_id_column_name

    speaker_dataset = pd.read_csv(args.tsv_path, sep="|", on_bad_lines='skip')
    speaker_column = ' SPEAKER   ' 
    gender_column = '   GENDER   '
    speaker_dataset = speaker_dataset.set_index(speaker_column)[gender_column]
    speaker_dataset = speaker_dataset.to_dict()
    
    def map_gender(speaker_ids):
        genders = [speaker_dataset[int(speaker)].strip() for speaker in speaker_ids]
        return {"gender": ["male" if g=="M" else "female" for g in genders]}
    
    dataset = dataset.map(map_gender, batched=True, batch_size=128, input_columns=speaker_id_column_name, num_proc=args.cpu_num_workers)

    
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
    
