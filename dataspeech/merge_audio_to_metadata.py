import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from multiprocess import set_start_method
import argparse



if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Repo id.")
    parser.add_argument("metadata_dataset_name", type=str, help="Repo id.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--dump_folder_path", default=None, type=str, help="If specified, save the dasaset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers.")
    parser.add_argument("--strategy", default="concatenate", type=str, help="For now only concatenate.")
    parser.add_argument("--id_column_name", default="id", type=str, help="For now only concatenate.") # TODO
    parser.add_argument("--columns_to_drop", default=None, type=str, help="Column names to drop in the metadataset. If some columns are duplicates. Separated by '+'. ")
    

    args = parser.parse_args()
    
    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)
        
    if args.configuration:
        metadata_dataset = load_dataset(args.metadata_dataset_name, args.configuration)
    else:
        metadata_dataset = load_dataset(args.metadata_dataset_name)

    columns_to_drop = None
    if args.columns_to_drop is not None:
        columns_to_drop = args.columns_to_drop.split("+")
        metadata_dataset = metadata_dataset.remove_columns(columns_to_drop)
    
    # TODO: for now suppose that they've kept the same ordering
    for split in dataset:
        if split in metadata_dataset:
            dataset[split] = concatenate_datasets([dataset[split], metadata_dataset[split].rename_column(args.id_column_name, f"metadata_{args.id_column_name}")], axis=1)
        else:
            raise ValueError(f"Metadataset don't have the same split {split} than dataset")
        
        if len(dataset[split].filter(lambda id1, id2: id1!=id2, input_columns=[args.id_column_name, f"metadata_{args.id_column_name}"])) != 0:
            raise ValueError(f"Concatenate didn't work. Some ids don't correspond on split {split}")
    

    if args.dump_folder_path:
        dataset.save_to_disk(args.dump_folder_path)
    if args.repo_id:
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
    