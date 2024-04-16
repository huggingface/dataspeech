import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from multiprocess import set_start_method
import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
import json

SPEAKER_RATE_BINS = ["very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast"]
SNR_BINS = ["very noisy", "quite noisy", "slightly noisy", "moderate ambient sound", "slightly clear", "quite clear", "very clear"]
REVERBERATION_BINS = ["very roomy sounding", "quite roomy sounding", "slightly roomy sounding", "moderate reverberation", "slightly confined sounding", "quite confined sounding", "very confined sounding"]
UTTERANCE_LEVEL_STD = ["very monotone", "quite monotone", "slightly monotone", "moderate intonation", "slightly expressive", "quite expressive", "very expressive"]

# this one is supposed to be apply to speaker-level mean pitch, and relative to gender
SPEAKER_LEVEL_PITCH_BINS = ["very low pitch", "quite low pitch", "slightly low pitch", "moderate pitch", "slightly high pitch", "quite high pitch", "very high pitch"]


def visualize_bins_to_text(values_1, values_2, name_1, name_2, text_bins, save_dir, output_column_name, default_bins=100, lower_range=None):
    # Save both histograms into a single figure
    fig, axs = plt.subplots(2, figsize=(8,6), sharex=True)
    
    # Plot histogram and vertical lines for subplot 1
    axs[0].hist(values_1, bins=default_bins, color='blue', alpha=0.7)
    _, bin_edges1 = np.histogram(values_1, bins=len(text_bins), range=(lower_range, values_1.max()) if lower_range else None)
    for edge in bin_edges1:
        axs[0].axvline(x=edge, color='red', linestyle='--', linewidth=1)


    # Plot histogram and vertical lines for subplot 2
    axs[1].hist(values_2, bins=50, color='green', alpha=0.7)
    _, bin_edges2 = np.histogram(values_2, bins=len(text_bins), range=(lower_range, values_2.max()) if lower_range else None)
    for edge in bin_edges2:
        axs[1].axvline(x=edge, color='red', linestyle='--', linewidth=1)

    # Add labels and title
    axs[0].set_title(name_1)
    axs[1].set_title(name_2)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].set_ylabel('Frequency')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel(f'{output_column_name}')

    # Adjust layout
    plt.tight_layout()

    filename = f"{output_column_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    print(f"Plots saved at '{filename}'!")

def bins_to_text(dataset, text_bins, column_name, output_column_name, leading_split_for_bins="train", batch_size = 4, num_workers = 1, std_tolerance=5, save_dir=None, only_save_plot=False, lower_range=None):
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
    if std_tolerance is not None:
        filtered_values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]

    if save_dir is not None:
        visualize_bins_to_text(values, filtered_values, "Before filtering", "After filtering", text_bins, save_dir, output_column_name, lower_range)
        
    # speaking_rate can easily have outliers
    if save_dir is not None and output_column_name=="speaking_rate":
        visualize_bins_to_text(filtered_values, filtered_values, "After filtering", "After filtering", text_bins, save_dir, f"{output_column_name}_after_filtering", lower_range)

    if only_save_plot:
        return dataset
    
    values = filtered_values
    hist, bin_edges = np.histogram(values, bins = len(text_bins), range=(lower_range, values.max()) if lower_range else None)

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

def speaker_level_relative_to_gender(dataset, text_bins, speaker_column_name, gender_column_name, column_name, output_column_name, batch_size = 4, num_workers=1, std_tolerance=None, save_dir=None, only_save_plot=False):
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
    if save_dir is not None:
        save_dict = {}
        save_dict_afer_filtering = {}
    for category in ["male", "female"]:
        values = dataframe[dataframe[gender_column_name] == category][column_name]
        values = np.array(values)
        if save_dir is not None:
            save_dict[category] = values
        if std_tolerance is not None:
            # filter out outliers
            values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]
            if save_dir is not None:
                save_dict_afer_filtering[category] = values
        bin_edges[category] = np.histogram(values, len(text_bins))[1]
        
    if save_dir is not None:
        visualize_bins_to_text(save_dict["male"], save_dict["female"], "Male distribution", "Female distribution", text_bins, save_dir, output_column_name)
        if std_tolerance is not None:
            visualize_bins_to_text(save_dict_afer_filtering["male"], save_dict_afer_filtering["female"], "Male distribution", "Female distribution", text_bins, save_dir, f"{output_column_name}_after_filtering")

    if only_save_plot:
        return dataset
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
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset(s) on disk. If multiple datasets, paths have to be separated by `+`.")
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
    parser.add_argument("--only_save_plot", default=False, type=bool, help="If `True` and `--plot_directory` is specified, will only compute plot.")
    parser.add_argument("--text_bins_path", default=None, type=str, help="If specified, points to a JSON file which contains the text bins desired. By default, will use default bins.")
    parser.add_argument("--snr_lower_range", default=50, type=float, help="The lower range of the SNR bins")

    args = parser.parse_args()
    
    if args.plot_directory is None and args.only_save_plot:
        raise ValueError("`only_save_plot=True` but `plot_directory` is not specified. Please give a path to the directory where you want the plot to be saved.")
    
    text_bins_dict = {}
    if args.text_bins_path:
        with open(args.text_bins_path) as json_file:
            text_bins_dict = json.load(json_file)

    speaker_level_pitch_bins = text_bins_dict.get("speaker_level_pitch_bins", SPEAKER_LEVEL_PITCH_BINS)
    speaker_rate_bins = text_bins_dict.get("speaker_rate_bins", SPEAKER_RATE_BINS)
    snr_bins = text_bins_dict.get("snr_bins", SNR_BINS)
    reverberation_bins = text_bins_dict.get("reverberation_bins", REVERBERATION_BINS)
    utterance_level_std = text_bins_dict.get("utterance_level_std", UTTERANCE_LEVEL_STD)
    
    output_dirs = [args.output_dir] if args.output_dir is not None else None
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

            if args.output_dir is not None:
                output_dirs = args.output_dir.split("+")
                if len(dataset_names) != len(output_dirs):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(output_dirs)} local paths on which to save the datasets spotted")
            
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

            if args.output_dir is not None:
                output_dirs = args.output_dir.split("+")
                if len(dataset_names) != len(output_dirs):
                    raise ValueError(f"There are {len(dataset_names)} datasets spotted but {len(output_dirs)} local paths on which to save the datasets spotted")
            
            dataset = []
            for dataset_name, dataset_config in zip(dataset_names):
                tmp_dataset = load_dataset(dataset_name)
                dataset.append(tmp_dataset)

        else:
            dataset = [load_dataset(args.dataset_name)]

    if args.plot_directory:
        Path(args.plot_directory).mkdir(parents=True, exist_ok=True)
    
    dataset = speaker_level_relative_to_gender(dataset, speaker_level_pitch_bins, args.speaker_id_column_name, args.gender_column_name, "utterance_pitch_mean", "pitch", batch_size=args.batch_size, num_workers=args.cpu_num_workers, std_tolerance=args.pitch_std_tolerance, save_dir=args.plot_directory, only_save_plot=args.only_save_plot)
    dataset = bins_to_text(dataset, speaker_rate_bins, "speaking_rate", "speaking_rate", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.speaking_rate_std_tolerance, save_dir=args.plot_directory, only_save_plot=args.only_save_plot)
    dataset = bins_to_text(dataset, snr_bins, "snr", "noise", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.snr_std_tolerance, save_dir=args.plot_directory, only_save_plot=args.only_save_plot, lower_range=args.snr_lower_range)
    dataset = bins_to_text(dataset, reverberation_bins, "c50", "reverberation", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.reverberation_std_tolerance, save_dir=args.plot_directory, only_save_plot=args.only_save_plot)
    dataset = bins_to_text(dataset, utterance_level_std, "utterance_pitch_std", "speech_monotony", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=args.leading_split_for_bins, std_tolerance=args.speech_monotony_std_tolerance, save_dir=args.plot_directory, only_save_plot=args.only_save_plot)

    if not args.only_save_plot:
        if args.output_dir:
            for output_dir, df in zip(output_dirs, dataset):
                df.save_to_disk(output_dir)
        if args.repo_id:
            for i, (repo_id, df) in enumerate(zip(repo_ids, dataset)):
                if args.configuration:
                    df.push_to_hub(repo_id, dataset_configs[i])
                else:
                    df.push_to_hub(repo_id)