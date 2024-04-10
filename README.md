# Data-Speech

Data-Speech is a suite of utility scripts designed to tag speech datasets. 

Its aim is to provide a simple, clean codebase for applying audio transformations (or annotations) that may be requested as part of the development of speech-based AI models, such as text-to-speech engines.

Its primary use is to reproduce the annotation method from Dan Lyth and Simon King's research paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912), that labels various speaker characteristics with natural language descriptions.

Applying these tools allows us to prepare and release tagged versions of [LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_tags_tagged_10k_generated), and a 10K hours subset of [the English version of MLS](https://huggingface.co/datasets/parler-tts/mls-eng-10k-tags_tagged_10k_generated).

This repository is designed to accompany the [Parler-TTS library](https://github.com/huggingface/parler-tts), which contains the inference and training code for Parler-TTS, a new family of high-quality text-to-speech models.

---------


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


## Set-up

You first need to clone this repository before installing requirements.

```sh
git clone git@github.com:ylacombe/dataspeech.git
cd dataspeech
pip install -r requirements.txt
```


## 1. Predict annotations

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


## 2. Map continuous annotations to key-words

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

## 3. Generate natural language descriptions

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

As usual, we precise the dataset name and configuration we want to annotate. `model_name_or_path` should point to a `transformers` model for prompt annotation. You can find a list of such models [here](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=trending). Here, we used a version of Mistral's 7B model.

The folder [`examples/prompt_creation/`](examples/prompt_creation/) contains two more examples.


> [!TIP]
> Scripts from this library can also be used as a starting point for applying other models to other datasets from the [datasets library](https://huggingface.co/docs/datasets/v2.17.0/en/index) in a large-scale settings.
> 
> For example, `scripts/run_prompt_creation.py` can be adapted to perform large-scaled inference using other LLMs and prompts.


## To conclude

In the [`/examples`](/examples/) folder, we applied this recipe to both [10K hours of MLS Eng](https://huggingface.co/datasets/parler-tts/mls-eng-10k-tags_tagged_10k_generated) and [LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_tags_tagged_10k_generated). The resulting datasets were used to train [Parler-TTS](https://github.com/huggingface/parler-tts), a new text-to-speech model.

This recipe is both scalable and easily modifiable and will hopefully help the TTS research community explore new ways of conditionning speech synthesis. 


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
