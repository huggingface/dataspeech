# Data-Speech

Data-Speech is a suite of utility scripts designed to tag speech datasets. 

Its aim is to provide a simple, clean codebase for applying audio transformations (or annotations) that may be requested as part of the development of speech-based AI models, such as text-to-speech engines.

Its primary use is to reproduce the annotation method from Dan Lyth and Simon King's research paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912), that labels various speaker characteristics with natural language descriptions.

Applying these tools allows us to prepare and release tagged versions of [LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_tags_tagged_10k_generated), and a 10K hours subset of [the English version of MLS](https://huggingface.co/datasets/parler-tts/mls-eng-10k-tags_tagged_10k_generated).

This repository is designed to accompany the [Parler-TTS library](https://github.com/huggingface/parler-tts), which contains the inference and training code for Parler-TTS, a new family of high-quality text-to-speech models.

---------

## 📖 Quick Index
* [Requirements](#set-up)
* [Annotating datasets to fine-tune Parler-TTS](#annotating-datasets-to-fine-tune-parler-tts)
* [Annotating datasets from scratch](#annotating-datasets-from-scratch)
* [Using Data-Speech to filter your speech datasets](#using-data-speech-to-filter-your-speech-datasets)
* [❓ FAQ](#faq)


## Set-up

You first need to clone this repository before installing requirements.

```sh
git clone git@github.com:huggingface/dataspeech.git
cd dataspeech
pip install -r requirements.txt
```

## Annotating datasets to fine-tune Parler-TTS

In the following examples, we'll load 30 hours of audio data from the [Jenny TTS dataset](https://github.com/dioco-group/jenny-tts-dataset), a high-quality mono-speaker TTS dataset, from an Irish female speaker named Jenny.

The aim here is to create an annotated version of Jenny TTS, in order to fine-tune the [Parler-TTS v0.1 checkpoint](https://huggingface.co/parler-tts/parler_tts_mini_v0.1) on this dataset.

Thanks to a [script similar to what's described in the FAQ](#how-do-i-use-datasets-that-i-have-with-this-repository), we've uploaded the dataset to the HuggingFace hub, under the name [reach-vb/jenny_tts_dataset](https://huggingface.co/datasets/reach-vb/jenny_tts_dataset).

Feel free to follow the link above to listen to some samples of the Jenny TTS dataset thanks to the hub viewer.

> [!IMPORTANT]
> Refer to the section [Annotating datasets from scratch](#annotating-datasets-from-scratch) for more detailed explanations of what's going on under-the-hood.

We'll:
1. Annotate the Jenny dataset with continuous variables that measures the speech characteristics
2. Map those annotations to text bins that characterize the speech characteristics.
3. Create natural language descriptions from those text bins

### 1. Annotate the Jenny dataset

We'll use [`main.py`](main.py) to get the following continuous variables:
    - Speaking rate `(nb_phonemes / utterance_length)`
    - Signal-to-noise ratio (SNR)
    - Reverberation
    - Speech monotony

```sh
python main.py "reach-vb/jenny_tts_dataset" \
  --configuration "default" \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --rename_column \
  --repo_id "jenny-tts-tags"
```

Note that the script will be faster if you have GPUs at your disposal. It will automatically scale-up to every GPUs available in your environnement.

The resulting dataset will be pushed to the HuggingFace hub under your HuggingFace handle. Mine was push to [ylacombe/jenny-tts-tags](https://huggingface.co/datasets/ylacombe/jenny-tts-tags).

### 2. Map annotations to text bins

Since the ultimate goal here is to fine-tune the [Parler-TTS v0.1 checkpoint](https://huggingface.co/parler-tts/parler_tts_mini_v0.1) on the Jenny dataset, we want to stay consistent with the text bins of the datasets on which the latter model was trained.

This is easy to do thanks to the following command:

```sh
python ./scripts/metadata_to_text.py \
    "ylacombe/jenny-tts-tags" \
    --repo_id "jenny-tts-tags" \
    --configuration "default" \
    --cpu_num_workers "8" \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation
```

Thanks to [`v01_bin_edges.json`](/examples/tags_to_annotations/v01_bin_edges.json), we don't need to recompute bins from scratch and the above script takes a few seconds.

The resulting dataset will be pushed to the HuggingFace hub under your HuggingFace handle. Mine was push to [ylacombe/jenny-tts-tags](https://huggingface.co/datasets/ylacombe/jenny-tts-tags).

You can notice that text bins such as `slightly noisy`, `quite monotone` have been added to the samples.

### 3. Create natural language descriptions from those text bins

Now that we have text bins associated to the Jenny dataset, the next step is to create natural language descriptions out of the few created features.

Here, we decided to create prompts that use the name `Jenny`, prompts that'll look like the following:
`In a very expressive voice, Jenny pronounces her words incredibly slowly. There's some background noise in this room with a bit of echo'`

This step generally demands more resources and times and should use one or many GPUs.

[`run_prompt_creation_jenny.sh`](examples/prompt_creation/run_prompt_creation_jenny.sh) indicates how to run it on LibriTTS-R:

```sh
python ./scripts/run_prompt_creation_single_speaker.py \
  --speaker_name "Jenny" \
  --dataset_name "ylacombe/jenny-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 32 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 8 \
  --output_dir "./tmp_jenny" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "jenny-tts-10k-tagged" \
  --preprocessing_num_workers 16
```

As usual, we precise the dataset name and configuration we want to annotate. `model_name_or_path` should point to a `transformers` model for prompt annotation. You can find a list of such models [here](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=trending). Here, we used a version of Mistral's 7B model.

> [!NOTE]
> If you want to use this on a multi-speaker dataset, you'd have to adapt the logic of the script.


## Annotating datasets from scratch

In the following examples, we'll load 1,000 hours of labelled audio data from the [LibriTTS-R dataset](https://huggingface.co/datasets/blabble-io/libritts_r) and add annotations using the dataspeech library. The resulting dataset is complete with discrete annotation tags, as well as a coherent audio
description of the spoken audio characteristics.


There are 3 steps to be completed in order to generate annotations:
1. [Annotate the speech dataset](#predict-annotations) to get the following continuous variables:
    - Speaking rate `(nb_phonemes / utterance_length)`
    - Signal-to-noise ratio (SNR)
    - Reverberation
    - Pitch estimation
2. [Map the previous annotations categorical to discrete keywords bins](#map-continuous-annotations-to-key-words)
3. [Create natural language descriptions from a set of keywords](#generate-natural-language-descriptions)


### 1. Predict annotations

For the time being, [`main.py`](main.py) can be used to generate speaking rate, SNR, reverberation and pitch estimation. 

To use it, you need a dataset from the [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, either locally or on the [hub](https://huggingface.co/datasets).


```sh
python main.py "blabble-io/libritts_r" \
--configuration "dev" \
--output_dir ./tmp_libritts_r_dev/ \
--text_column_name "text_normalized" \
--audio_column_name "audio" \
--cpu_num_workers 8 \
--num_workers_per_gpu 4 \
--rename_column \
```

Here, we've used 8 processes for operations that don't use GPUs, namely to compute the speaking rate. If GPUs were present in the environnement, `num_workers_per_gpu` precises the number of processes per GPUs for the operations that can be computed on GPUs - namely pitch, SNR and reverberation estimation.

You can learn more about the arguments you can pass to `main.py` by passing:

```sh
python main.py --help
```

In [`/examples/tagging/run_main_1k.sh`](/examples/run_main_1k.sh), we scaled up the initial command line to the whole dataset. Note that we've used the `repo_id` argument to push the dataset to the hub, resulting in [this dataset](https://huggingface.co/datasets/ylacombe/libritts_r_tags).

The dataset viewer gives an idea of what has been done, namely:
- new columns were added:
    - `utterance_pitch_std`: Gives a measure of the standard deviation of pitch in the utterance.
    - `utterance_pitch_mean`: Gives a measure of average pitch in the utterance.
    - `snr`: Speech-to-noise ratio
    - `c50`: Reverberation estimation
    - `speaking_rate`
    - `phonemes`: which was used to compute the speaking rate
- the audio column was removed - this is especially useful when dealing with big datasets, as writing and pushing audio data can become a bottleneck.

![image](https://github.com/ylacombe/dataspeech/assets/52246514/f422a728-f2af-4c8f-bf2a-65c6722bc0c6)


### 2. Map continuous annotations to key-words

The next step is to map the continuous annotations from the previous steps to key-words. To do so, continous annotations are mapped to categorical bins that are then associated to key-words. For example, the speaking rate can be associated to 7 text bins which are: `"very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast"`.

[`scripts/metadata_to_text.py`](/scripts/metadata_to_text.py) computes bins on aggregated statistics from multiple datasets:
- A speaker's pitch is calculated by averaging the pitches across its voice clips. The computed pitch estimator is then compared to speakers of the same gender to derive the pitch keyword of the speaker(very high-pitched to very low-pitched).
- The rest of the keywords are derived by [computing histograms](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) of the continuous variables over all training samples, from which the extreme values have been eliminated, and associating a keyword with each bin.


```sh
python ./scripts/metadata_to_text.py "ylacombe/libritts_r_tags+ylacombe/libritts_r_tags" \
--configuration "clean+other" \
--output_dir "./tmp_tts_clean+./tmp_tts_other" \
--cpu_num_workers "8" \
--leading_split_for_bins "train" \
--plot_directory "./plots/" \
```
Note how we've been able to pass different datasets with different configurations by separating the relevant arguments with `"+"`.

By passing `--repo_id parler-tts/libritts-r-tags-and-text+parler-tts/libritts-r-tags-and-text`, we pushed the resulting dataset to [this hub repository](https://huggingface.co/datasets/parler-tts/libritts-r-tags-and-text).

Note that this step is a bit more subtle than the previous one, as we generally want to collect a wide variety of speech data to compute accurate key-words. 

Indeed, some datasets, such as LibriTTS-R, collect data from only one or a few sources; for LibriTTS-R, these are audiobooks, and the process of collecting or processing the data can result in homogeneous data that has little variation. In the case of LibriTTS-R, the data has been cleaned to have little noise, little reverberation, and the audiobooks collected leaves little variety in intonation.

You can learn more about the arguments you can pass to `main.py` by passing:

```sh
python main.py --help
```

### 3. Generate natural language descriptions

Now that we have text bins associated to our datasets, the next step is to create natural language descriptions out of the few created features.

[`scripts/run_prompt_creation.py`](/scripts/run_prompt_creation.py) relies on [`accelerate`](https://huggingface.co/docs/accelerate/index) and [`transformers`](https://huggingface.co/docs/transformers/index) to generate natural language descriptions from LLMs. This step generally demands more resources and times and should use one or many GPUs.

[`examples/prompt_creation/run_prompt_creation_1k.sh`](examples/prompt_creation/run_prompt_creation_1k.sh) indicates how to run it on LibriTTS-R:

```sh
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "parler-tts/libritts-r-tags-and-text" \
  --dataset_config_name "clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "parler-tts/libritts-r-tags-and-text-generated"
```

As usual, we define the dataset name and configuration we want to annotate. `model_name_or_path` should point to a `transformers` model for prompt annotation. You can find a list of such models [here](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=trending). Here, we used a version of Mistral's 7B model. If you use a LLaMA or Gemma checkpoint, you can improve throughput by up to 1.5x by adding the flag `--torch_compile` to the arguments.

The folder [`examples/prompt_creation/`](examples/prompt_creation/) contains more examples.


> [!TIP]
> Scripts from this library can also be used as a starting point for applying other models to other datasets from the [datasets library](https://huggingface.co/docs/datasets/v2.17.0/en/index) in a large-scale settings.
> 
> For example, `scripts/run_prompt_creation.py` can be adapted to perform large-scaled inference using other LLMs and prompts.


### To conclude

In the [`/examples`](/examples/) folder, we applied this recipe to both [10K hours of MLS Eng](https://huggingface.co/datasets/parler-tts/mls-eng-10k-tags_tagged_10k_generated) and [LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_tags_tagged_10k_generated). The resulting datasets were used to train [Parler-TTS](https://github.com/huggingface/parler-tts), a new text-to-speech model.

This recipe is both scalable and easily modifiable and will hopefully help the TTS research community explore new ways of conditionning speech synthesis. 

## Using Data-Speech to filter your speech datasets

While the rest of the README explains how to use this repository to create text descriptions of speech utterances, Data-Speech can also be used to perform filtering on speech datasets.

For example, you can
1. Use the [`Predict annotations`](#1-predict-annotations) step to predict SNR and reverberation.
2. Filter your data sets to retain only the most qualitative samples.

You could also, to give more examples, filter on a certain pitch level (e.g only low-pitched voices), or a certain speech rate (e.g only fast speech).

## FAQ

### What kind of datasets do I need?

We rely on the [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, which is optimized for speed and efficiency, and is deeply integrated with the [HuggingFace Hub](https://huggingface.co/datasets) which allows easy sharing and loading.

In order to use this repository, you need a speech dataset from [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) with at least one audio column and a text transcription column.

### How do I use datasets that I have with this repository?

If you have a local dataset, and want to create a dataset from [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) to use Data-Speech, you can use the following recipes or refer to the [`dataset` docs](https://huggingface.co/docs/datasets/v2.17.0/en/index) for more complex use-cases.

1. You first need to create a csv file that contains the **full paths** to the audio. The column name for those audio files could be for example `audio`, but you can use whatever you want. You also need a column with the transcriptions of the audio, this column can be named `transcript` but you can use whatever you want.

2. Once you have this csv file, you can load it to a dataset like this:
```python
from datasets import DatasetDict

dataset = DatasetDict.from_csv({"train": PATH_TO_CSV_FILE})
```
3. You then need to convert the audio column name to [`Audio`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Audio) so that `datasets` understand that it deals with audio files.
```python
from datasets import Audio
dataset = dataset.cast_column("audio", Audio())
```
4. You can then [push the dataset to the hub](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub):
```python
dataset.push_to_hub(REPO_ID)
```

Note that you can make the dataset private by passing [`private=True`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub.private) to the [`push_to_hub`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub) method. Find other possible arguments [here](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).

When using Data-Speech, you can then use `REPO_ID` (replace this by the name you want here and above) as the dataset name.


## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively, for publishing such a promising and clear research paper: [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912).
- and the many libraries used, namely [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [brouhaha](https://github.com/marianne-m/brouhaha-vad/blob/main/README.md), [penn](https://github.com/interactiveaudiolab/penn/blob/master/README.md), [g2p](https://github.com/Kyubyong/g2p), [accelerate](https://huggingface.co/docs/accelerate/en/index) and [transformers](https://huggingface.co/docs/transformers/index).

## Citation

If you found this repository useful, please consider citing this work and also the original Stability AI paper:

```
@misc{lacombe-etal-2024-dataspeech,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Data-Speech},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ylacombe/dataspeech}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Status
This library is still a WIP. Other utility scripts should come soon.

### TODOs
- [ ] Accent classification training script
- [ ] Accent classification inference script
- [ ] Better speaking rate estimation with long silence removal
- [ ] Better SNR estimation with other SNR models
- [ ] Add more annotation categories
- [ ] Multilingual speaking rate estimation

- [ ] (long term) Benchmark for best audio dataset format
- [ ] (long term) Compatibility with streaming
