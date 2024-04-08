import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from multiprocess import set_start_method
import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt

SPEAKER_RATE_BINS = ["very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast"]
SNR_BINS = ["very noisy", "quite noisy", "slightly noisy", "moderate ambient sound", "slightly clear", "quite clear", "very clear"]
REVERBERATION_BINS = ["very roomy sounding", "quite roomy sounding", "slightly roomy sounding", "moderate reverberation", "slightly confined sounding", "quite confined sounding", "very confined sounding"]
UTTERANCE_LEVEL_STD = ["very monotone", "quite monotone", "slightly monotone", "moderate intonation", "slightly expressive", "quite expressive", "very expressive"]

# this one is supposed to be apply to speaker-level mean pitch, and relative to gender
SPEAKER_LEVEL_PITCH_BINS = ["very low pitch", "quite low pitch", "slightly low pitch", "moderate pitch", "slightly high pitch", "quite high pitch", "very high pitch"]


def visualize_bins_to_text(hist, filtered_hist, text_bins, save_dir, output_column_name):
    # Save both histograms into a single figure
    plt.figure(figsize=(8,6))
    plt.suptitle('Before & After Standard Deviation Tolerance')
    axes = plt.gca().yaxis
    axes.grid(alpha=.3)


    plt.subplot(2, 1, 1)
    plt.bar(range(len(hist)), hist, color='blue', alpha=0.7)
    plt.title('Original Distribution')
    plt.yticks(fontsize=9)
    plt.yscale("log")
    plt.xticks([r for r in range(len(hist))], text_bins, fontsize=5)

    plt.subplot(2, 1, 2)
    plt.bar(range(len(filtered_hist)), filtered_hist, color='red', alpha=0.7)
    plt.title('Filtered Distribution')
    plt.yticks(fontsize=9)
    plt.yscale("log")
    plt.xticks([r for r in range(len(filtered_hist))], text_bins, fontsize=5)

    filename = f"{output_column_name}_before_after_sd_threshold.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    print(f"Plots saved at '{filename}'!")

def bins_to_text(dataset, text_bins, column_name, output_column_name, leading_split_for_bins="train", batch_size = 4, num_workers = 1, std_tolerance=5, save_dir=None):
    '''
    Compute bins of `column_name` from the splits `leading_split_for_bins` and apply text bins to every split.
    `leading_split_for_bins` can be a string or a list.
    '''
    values = []
    for df in dataset:
        for split in df:
            if leading_split_for_bins is None or leading_split_for_bins in split:
                values.extend(df[split][column_name])
    
    # filter out outliers
    values = np.array(values)
    original_hist, _ = np.histogram(values, bins = len(text_bins))
    if std_tolerance is not None:
        values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]

    hist, bin_edges = np.histogram(values, bins = len(text_bins))
    if save_dir is not None:
        visualize_bins_to_text(original_hist, hist, text_bins, save_dir, output_column_name)
    
    def batch_association(batch):
        index_bins = np.searchsorted(bin_edges, batch, side="left")
        # do min(max(...)) when values are outside of the main bins
        # it happens when value = min or max or have been filtered out from bins computation
        batch_bins = [text_bins[min(max(i-1, 0), len(text_bins)-1)] for i in index_bins]
        return {
            output_column_name: batch_bins
        }
    
    dataset = [df.map(batch_association, batched=True, batch_size=batch_size, input_columns=[column_name], num_proc=num_workers) for df in dataset]
    return dataset

def speaker_level_relative_to_gender(dataset, text_bins, speaker_column_name, gender_column_name, column_name, output_column_name, batch_size = 4, num_workers=1, std_tolerance=None):
    '''
    Computes mean values on a speaker level and computes bins on top relative to the gender column name.
    Then associate a text bin to the column.
    This time, doesn't use leading_split_for_bins, computes it for all. Could probably be optimized
    '''
    list_data = []
    for df in dataset:
        for split in df:
            panda_data = df[split].remove_columns([col for col in df[split].column_names if col not in {speaker_column_name, column_name, gender_column_name}]).to_pandas()
            list_data.append(panda_data)
        
    dataframe = pd.concat(list_data, ignore_index=True)
    dataframe = dataframe.groupby(speaker_column_name).agg({column_name: "mean", gender_column_name: "first"})
    
    bin_edges = {}
    for category in ["male", "female"]:
        values = dataframe[dataframe[gender_column_name] == category][column_name]
        values = np.array(values)
        if std_tolerance is not None:
            # filter out outliers
            values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]
        bin_edges[category] = np.histogram(values, len(text_bins))[1]
        
    # 
    speaker_id_to_bins = dataframe.apply(lambda x: np.searchsorted(bin_edges[x[gender_column_name]], x[column_name]), axis=1).to_dict()
        
    def batch_association(batch):
        index_bins = [speaker_id_to_bins[speaker] for speaker in batch]
        # do min(max(...)) when values are outside of the main bins
        # it happens when value = min or max or have been filtered out from bins computation
        batch_bins = [text_bins[min(max(i-1, 0), len(text_bins)-1)] for i in index_bins]
        return {
            output_column_name: batch_bins
        }
        
    
    dataset = [df.map(batch_association, batched=True, input_columns=[speaker_column_name], batch_size=batch_size, num_proc=num_workers) for df in dataset]
    return dataset

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset(s). If multiple datasets, names have to be separated by `+`.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration(s) to use (or configuration separated by +).")
    parser.add_argument("--dump_folder_path", default=None, type=str, help="If specified, save the dataset(s) on disk. If multiple datasets, paths have to be separated by `+`.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset(s) to the hub. If multiple datasets, names have to be separated by `+`.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size in `Dataset.map` operations. https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Dataset.map")
    parser.add_argument("--speaker_id_column_name", default="speaker_id", type=str, help="Speaker id column name.")
    parser.add_argument("--gender_column_name", default="gender", type=str, help="Gender column name.")
    parser.add_argument("--pitch_std_tolerance", default=2., type=float, help="Standard deviation tolerance for pitch estimation. Any value that is outside mean ± std * tolerance is discared.")
    parser.add_argument("--speaking_rate_std_tolerance", default=4., type=float, help="Standard deviation tolerance for speaking rate estimation. Any value that is outside mean ± std * tolerance is discared.")
    parser.add_argument("--snr_std_tolerance", default=3.5, type=float, help="Standard deviation tolerance for SNR estimation. Any value that is outside mean ± std * tolerance is discared.")
    parser.add_argument("--reverberation_std_tolerance", default=4, type=float, help="Standard deviation tolerance for reverberation estimation. Any value that is outside mean ± std * tolerance is discared.")
    parser.add_argument("--speech_monotony_std_tolerance", default=4, type=float, help="Standard deviation tolerance for speech monotony estimation. Any value that is outside mean ± std * tolerance is discared.")
    parser.add_argument("--leading_split_for_bins", default=None, type=str, help="If specified, will use every split that contains it to compute statistics. If not specified, will use every split.")
    parser.add_argument("--plot_directory", default=None, type=str, help="If specified, will save visualizing plots to this directory.")

    args = parser.parse_args()
    
    dump_folder_paths = [args.dump_folder_path] if args.dump_folder_path is not None else None
    repo_ids = [args.repo_id] if args.repo_id is not None else None
    if args.configuration:
        if "+" in args.dataset_name:
            dataset_names = args.dataset_name.split("+")
            dataset_configs = args.configuration.split("+")
            if len(dataset_names) != len(dataset_configs):
                raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(dataset_configs)} configuration spotted")
            
            if args.repo_id is not None:
                repo_ids = args.repo_id.split("+")
                if len(dataset_names) != len(repo_ids):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(repo_ids)} repository ids spotted")

            if args.dump_folder_path is not None:
                dump_folder_paths = args.dump_folder_path.split("+")
                if len(dataset_names) != len(dump_folder_paths):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(dump_folder_paths)} local paths on which to save the datasets spotted")
            
            dataset = []
            for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
                tmp_dataset = load_dataset(dataset_name, dataset_config)
                dataset.append(tmp_dataset)
        else:
            dataset = [load_dataset(args.dataset_name, args.configuration)]
            dataset_configs = [args.configuration]
    else:
        if "+" in args.dataset_name:
            dataset_names = args.dataset_name.split("+")
            if args.repo_id is not None:
                repo_ids = args.repo_id.split("+")
                if len(dataset_names) != len(repo_ids):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(repo_ids)} repository ids spotted")

            if args.dump_folder_path is not None:
                dump_folder_paths = args.dump_folder_path.split("+")
                if len(dataset_names) != len(dump_folder_paths):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(dump_folder_paths)} local paths on which to save the datasets spotted")
            
            dataset = []
            for dataset_name, dataset_config in zip(dataset_names):
                tmp_dataset = load_dataset(dataset_name)
                dataset.append(tmp_dataset)

        else:
            dataset = [load_dataset(args.dataset_name)]

    if args.plot_directory:
        Path(args.plot_directory).mkdir(parents=True, exist_ok=True)
    
    dataset = speaker_level_relative_to_gender(dataset, SPEAKER_LEVEL_PITCH_BINS, args.speaker_id_column_name, args.gender_column_name, "utterance_pitch_mean", "pitch", batch_size=args.batch_size, num_workers=args.cpu_num_workers, std_tolerance=args.pitch_std_tolerance)
    dataset = bins_to_text(dataset, SPEAKER_RATE_BINS, "speaking_rate", "speaking_rate", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.speaking_rate_std_tolerance, save_dir=args.plot_directory)
    dataset = bins_to_text(dataset, SNR_BINS, "snr", "noise", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.snr_std_tolerance, save_dir=args.plot_directory)
    dataset = bins_to_text(dataset, REVERBERATION_BINS, "c50", "reverberation", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.reverberation_std_tolerance, save_dir=args.plot_directory)
    dataset = bins_to_text(dataset, UTTERANCE_LEVEL_STD, "utterance_pitch_std", "speech_monotony", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.speech_monotony_std_tolerance, save_dir=args.plot_directory)

    if args.dump_folder_path:
        for dump_folder_path, df in zip(dump_folder_paths, dataset):
            df.save_to_disk(dump_folder_path)
    if args.repo_id:
        for i, (repo_id, df) in enumerate(zip(repo_ids, dataset)):
            if args.configuration:
                df.push_to_hub(repo_id, dataset_configs[i])
            else:
                df.push_to_hub(repo_id)