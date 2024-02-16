import numpy as np
import pandas as pd
from datasets import load_dataset
from multiprocess import set_start_method
import argparse


SPEAKER_RATE_BINS = ["very slowly", "quite slowly", "fairly slowly", "moderate speed", "fairly fast", "quite fast", "very fast"]
SNR_BINS = ["very noisy", "quite noisy", "fairly noisy", "moderate ambient sound", "fairly quiet", "quite quiet", "very quiet"]
REVERBERATION_BINS = ["very roomy sounding", "quite roomy sounding", "fairly roomy sounding", "moderate reverberation", "fairly confined sounding", "quite confined sounding", "very confined sounding"]
UTTERANCE_LEVEL_STD = ["very monotone", "quite monotone", "fairly monotone", "moderate intonation", "fairly expressive", "quite expressive", "very expressive"]

# this one is supposed to be apply to speaker-level mean pitch, and relative to gender
SPEAKER_LEVEL_PITCH_BINS = ["very low pitch", "quite low pitch", "fairly low pitch", "moderate pitch", "fairly high pitch", "quite high pitch", "very high pitch"]


def bins_to_text(dataset, text_bins, column_name, output_column_name, leading_split_for_bins="train", batch_size = 4, num_workers = 1, std_tolerance=5):
    '''
    Compute bins of `column_name` from the splits `leading_split_for_bins` and apply text bins to every split.
    `leading_split_for_bins` can be a string or a list.
    '''
    if isinstance(leading_split_for_bins, str):
        leading_split_for_bins = [leading_split_for_bins]
        
    values = dataset[leading_split_for_bins[0]][column_name]
    for split in leading_split_for_bins[1:]:
        values.extend(dataset[split][column_name])

    # filter out outliers
    values = np.array(values)
    if std_tolerance is not None:
        values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]

    hist, bin_edges = np.histogram(values, bins = len(text_bins))
    
    def batch_association(batch):
        index_bins = np.searchsorted(bin_edges, batch, side="left")
        # do min(max(...)) when values are outside of the main bins
        # it happens when value = min or max or have been filtered out from bins computation
        batch_bins = [text_bins[min(max(i-1, 0), len(text_bins)-1)] for i in index_bins]
        return {
            output_column_name: batch_bins
        }
        
    dataset = dataset.map(batch_association, batched=True, batch_size=batch_size, input_columns=[column_name], num_proc=num_workers)
    return dataset

def do_mean_values_per_speaker(dataset, speaker_column_name, column_names_to_mean, batch_size = 4, num_workers = 1):
    '''
    Compute mean values per speaker and apply it to dataset. Won't be used here after giving it some thoughts.
    '''
    list_data = []
    for split in dataset.keys():
        panda_data = dataset[split].remove_columns([col for col in dataset[split].column_names if col !=speaker_column_name and col not in column_names_to_mean]).to_pandas()
        list_data.append(panda_data)
    
    dataframe = pd.concat(list_data, ignore_index=True)
    dataframe = dataframe.groupby(speaker_column_name).mean().to_dict()
    
    def columns_to_mean(speaker_ids):
        batch = {}
        for col in column_names_to_mean:
            batch[col] = [dataframe[col][speaker_id] for speaker_id in speaker_ids]
        
        return batch
    
    dataset = dataset.map(columns_to_mean, batched=True, batch_size=batch_size, input_columns = [speaker_column_name], num_proc=num_workers)
    return dataset
    

def speaker_level_relative_to_gender(dataset, text_bins, speaker_column_name, gender_column_name, column_name, output_column_name, batch_size = 4, num_workers=1, std_tolerance=None):
    '''
    Computes mean values on a speaker level and computes bins on top relative to the gender column name.
    Then associate a text bin to the column.
    This time, doesn't use leading_split_for_bins, computes it for all. Could probably be optimized
    '''
    list_data = []
    for split in dataset.keys():
        panda_data = dataset[split].remove_columns([col for col in dataset[split].column_names if col not in {speaker_column_name, column_name, gender_column_name}]).to_pandas()
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
        
    
    dataset = dataset.map(batch_association, batched=True, input_columns=[speaker_column_name], batch_size=batch_size, num_proc=num_workers)
    return dataset

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Repo id.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--dump_folder_path", default=None, type=str, help="If specified, save the dasaset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size when relevant") # TODO: better description

    args = parser.parse_args()
    
    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)



    dataset = speaker_level_relative_to_gender(dataset, SPEAKER_LEVEL_PITCH_BINS, "speaker_id", "gender", "utterance_pitch_mean", "pitch", batch_size=args.batch_size, num_workers=args.cpu_num_workers, std_tolerance=3)
    dataset = bins_to_text(dataset, SPEAKER_RATE_BINS, "speaking_rate", "speaking_rate", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=["train.clean.100", "train.clean.360"], std_tolerance=5)
    dataset = bins_to_text(dataset, SNR_BINS, "snr", "noise", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=["train.clean.100", "train.clean.360"], std_tolerance=10)
    dataset = bins_to_text(dataset, REVERBERATION_BINS, "c50", "reverberation", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=["train.clean.100", "train.clean.360"], std_tolerance=10)
    dataset = bins_to_text(dataset, UTTERANCE_LEVEL_STD, "utterance_pitch_std", "speech_monotony", batch_size=args.batch_size, num_workers=args.cpu_num_workers, leading_split_for_bins=["train.clean.100", "train.clean.360"], std_tolerance=5)

    if args.dump_folder_path:
        dataset.save_to_disk(args.dump_folder_path)
    if args.repo_id:
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
    