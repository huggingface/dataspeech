from datasets import load_dataset
from multiprocess import set_start_method
import pandas as pd
import argparse
from os import listdir
import os


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Repo id or local path.")
    parser.add_argument("bad_samples_folder", default=None, type=str, help="Path to LibriTTS-R bad folder samples.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dasaset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--speaker_id_column_name", default="speaker_id", type=str, help="Speaker id column name.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")

    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)
        
    speaker_id_column_name = args.speaker_id_column_name
    
    # speakers to exclude because of mixed gender detection
    # cf: https://github.com/line/LibriTTS-P/blob/main/data/excluded_spk_list.txt
    speakers_to_remove = {2074, 4455, 6032, 3546, 2262, 8097, 1734, 3793, 8295}
    
    def filter_speakers(speaker, speakers_to_remove):
        return int(speaker) not in speakers_to_remove 

    print(dataset)
    dataset = dataset.filter(filter_speakers, input_columns=speaker_id_column_name, num_proc=args.cpu_num_workers, fn_kwargs={"speakers_to_remove": speakers_to_remove})
    print(dataset)
    
    bad_samples_txt_files = [os.path.join(args.bad_samples_folder, f) for f in listdir(args.bad_samples_folder) if "bad_sample" in f] 

    samples_to_filter = set()
    for txt_file in bad_samples_txt_files:
        with open(txt_file, 'r') as file:
            for line in file:
                line = line.strip().split("/")[-1].split(".")[0]

                samples_to_filter.add(line)

    print(len(samples_to_filter))
    def filter_samples(id, samples_to_filter):
        return id not in samples_to_filter 
    dataset = dataset.filter(filter_samples, input_columns="id", num_proc=args.cpu_num_workers, fn_kwargs={"samples_to_filter": samples_to_filter})

    print(dataset)
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
    
