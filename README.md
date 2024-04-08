# Data-Speech

Data-Speech is a suite of utility scripts designed to tag speech datasets. 

Its aim is to provide a simple, clean codebase for applying audio transformations (or annotations) that may be requested as part of the development of speech-based AI models, such as text-to-speech engines.

Its primary use is to reproduce the annotation method from Dan Lyth and Simon King's research paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912), that labels various speaker characteristics with natural language descriptions.

Applying these tools allows us to prepare and release tagged versions of [LibriTTS-R](https://huggingface.co/datasets/blabble-io/libritts_r). and a 10K hours subset of [the English version of MLS](https://www.openslr.org/94/). Datasets can be found [here](https://huggingface.co/parler-tts).  # TODO(YL): I would replace the hyperlinks here with the direct hyper-links to the tagged dataset on the Hub


---------

## Set-up

You first need to clone this repository before installing requirements.

```sh
git clone git@github.com:ylacombe/dataspeech.git
cd dataspeech
pip install -r requirements.txt
```

## Use-cases

Current use-cases covers:
- [Annotate a speech dataset](#predict-annotations) to get the following continuous variables:
    - Speaking rate `(nb_phonemes / utterance_length)`
    - Signal-to-noise ratio (SNR)
    - Reverberation
    - Pitch estimation
- [Map the previous annotations categorical to discrete keywords bins](#map-continuous-annotations-to-key-words)
- [Create natural language descriptions from a set of keywords](#generate-natural-language-descriptions)

In the following examples, we'll load 1,000 hours of labelled audio data from the Libri TTS-R dataset and add annotations 
using the dataspeech library. The resulting dataset is complete with discrete annotation tags, as well as a coherent audio
description of the spoken audio characteristics.

> [!TIP]
> Scripts from this library can also be used as a starting point for applying other models to datasets from the [datasets library](https://huggingface.co/docs/datasets/v2.17.0/en/index) in a multi-machine configuration.
> 
> For example, `scripts/run_prompt_creation.py` can be adapted to perform large-scaled inference using other LLMs and prompts.

### 1. Predict annotations

For the time being, [`main.py`](main.py) can be used to generate speaking rate, SNR, reverberation and pitch estimation.

To use it, you need a dataset from the [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, either locally or on the [hub](https://huggingface.co/datasets).

For example, let's use it on the [450-samples midlands male split of the English dialect dataset](https://huggingface.co/datasets/ylacombe/english_dialects/viewer/midlands_male):

```sh
python main.py "ylacombe/english_dialects" \
--configuration "midlands_male" \
--dump_folder_path ./tmp_english_midlands/ \
--text_column_name "text" \
--audio_column_name "audio" \
--cpu_num_workers 8 \
--num_workers_per_gpu 4 \
```

Here, we've used 8 processes for operations that don't use GPUs, namely to compute the speaking rate. If GPUs were present in the environnement, `num_workers_per_gpu` precises the number of processes per GPUs for the operations that can be computed on GPUs - namely pitch, SNR and reverberation estimation.

You can learn more about the arguments you can pass to `main.py` by passing:

```sh
python main.py --help
```

In addition to the command line used as an example, we've used the `repo_id` argument to push the dataset to the hub, resulting in [this dataset](https://huggingface.co/datasets/ylacombe/example_process_dataset).

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

This step is more subtle than the previous one, as we generally want to collect a wide variety of speech data to compute accurate key-words.

Indeed, some datasets, such as LibriTTS-R, collect data from only one or a few sources; for LibriTTS-R, these are audiobooks, and the process of collecting or processing the data can result in homogeneous data that can vary. In the case of LibriTTS-R, the data has been cleaned to have little noise, little reverberation, and the audiobooks collected leaves little variety in intonation.

The solution here is to compute bins on aggregated statistics from multiple datasets, using [`scripts/metadata_to_text.py`](/scripts/metadata_to_text.py).
- A speaker's pitch is calculated by averaging the pitches across its voice clips. The computed pitch estimator is then compared to speakers of the same gender to derive the pitch keyword of the speaker(very high-pitched to very low-pitched).
- The rest of the keywords are derived by [computing histograms](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) of the continuous variables over all training samples, from which the extreme values have been eliminated, and associating a keyword with each bin.

The following command line uses the script on a 10K hours subset of MLS English and on the whole LibriTTS-R dataset.

```sh
python ./scripts/metadata_to_text.py "ylacombe/mls-eng-10k-tags+ylacombe/libritts_r_tags+ylacombe/libritts_r_tags" \
--configuration "default+clean+other" \
--dump_folder_path "./tmp_mls+./tmp_tts_clean+./tmp_tts_other" \
--cpu_num_workers "8" \
--leading_split_for_bins "train" \
--plot_directory "./plots/"
```

Note how we've been able to pass different datasets with different configurations by separating the relevant arguments with `"+"`.

You can learn more about the arguments you can pass to `main.py` by passing:

```sh
python main.py --help
```

Note that default tolerances for extreme values can also be modified by passing the desired value to the following arguments: `pitch_std_tolerance`, `speaking_rate_std_tolerance`, `snr_std_tolerance`, `reverberation_std_tolerance`, `speech_monotony_std_tolerance`.


### 3. Generate natural language descriptions

Now that we have text bins associated to our datasets, the next step is to create natural language descriptions out of the few created features.

[`scripts/run_prompt_creation.py`](/scripts/run_prompt_creation.py) relies on [`accelerate`](https://huggingface.co/docs/accelerate/index) and [`transformers`](https://huggingface.co/docs/transformers/index) to generate natural language descriptions from LLMs.

[`examples/prompt_creation/run_prompt_creation_dummy.sh`](examples/prompt_creation/run_prompt_creation_dummy.sh) contains a dummy example to get you ready:

```sh
python run_prompt_creation.py \
  --dataset_name "ylacombe/libritts_r_tags_and_text" \
  --dataset_config_name "clean" \
  --dataset_split_name "dev.clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 16 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 0 \
  --output_dir "./" \
  --load_in_4bit
```

As usual, we precise the dataset name and configuration we want to annotate. Here, you can also specify a single split, if necessary.
`model_name_or_path` should point to a `transformers` model for prompt annotation. You can find a list of such models [here](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=trending). Here, we used a version of Mistral's 7B model.

The folder [`examples/prompt_creation/`](examples/prompt_creation/) contains two more examples that scale the recipe to respectively 1K and 10K hours of data.

> [!CAUTION]
> This step generally demands more resources and times and should use one or many GPUs. 

## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

TODO(YL): bullet point acknowledgements (Stability paper, Dan Lyth, Datasets, + all other models that we use), add citation for Stability paper as well?

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
