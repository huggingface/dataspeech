

Don't forget to describe that you create another dataset

Don't forget to add examples script

Don't forget to add utility to merge datasets
            # TODO: probably should clean phonemes before using it for speaking_rate 


Don't forget to add accent classifier + LLM annotation

TODO: clean metadata
TODO add tools to visualize

TODO: shoutout to datasets and other libraries


TODO: Sanchit - what to do with run_dataset_concatenation.py -> better naming ?
https://github.com/sanchit-gandhi/stable-speech/blob/main/run_dataset_concatenation.py



# Data-Speech

Data-Speech is a suite of utility scripts designed to enrich audio speech datasets. 

Its aim is to provide a simple, clean code base for applying audio transformations or annotations that may be requested as part of the development of speech-based AI models.

Its primary use is to reproduce the annotation method from [Dan Lyth and Simon King's research paper (`Natural language guidance of high-fidelity text-to-speech with synthetic annotations`)](https://arxiv.org/abs/2402.01912) that allows to label various speaker characteristics with natural language description.

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
- [Annotate an audio speech dataset using `main.py`](#generate-annotations) to get the following continuous variables:
    - Speaking rate `(nb_phonemes / utterance_length)`
    - Speech-to-noise ratio (SNR)
    - Reverberation
    - Pitch estimation
- [Map the previous annotations categorical to discrete keywords bins using `scripts/metadata_to_text.py`](#map-continuous-annotations-to-key-words)
- [Create natural language descriptions from a set of keywords using `scripts/run_prompt_creation.py`](#generate-natural-language-descriptions)

Moreover, additional scripts cover other use-cases such as:
- [perform audio separation](#perform-audio-separation) using [demucs](TODO) in a multi-GPU settings
- add gender information to [MLS](TODO) and [LibriTTS-R](TODO).
- more to come...

> [!TIP]
> Scripts from this library can also be used as a starting point for applying other models to datasets from the [datasets library](TODO) in a multi-machine configuration.
> 
> For example, `scripts/run_prompt_creation.py` can be adapted to perform large-scaled inference using other LLMs and prompts.

### Generate annotations

For the time being, [`main.py`](main.py) can be used to generate speaking rate, SNR, reverberation and pitch estimation.

To use it, you need a dataset from the [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, either locally or on the [hub](https://huggingface.co/datasets).

For example, let's use it on the [450-samples midlands male split of the English dialect dataset](https://huggingface.co/datasets/ylacombe/english_dialects/viewer/midlands_male).

```sh
python main.py "ylacombe/english_dialects" \
--configuration "midlands_male" \
--dump_folder_path ./tmp_english_midlands/ \
--text_column_name "text" \
--audio_column_name "audio"
--cpu_num_workers 8 \
--num_workers_per_gpu 4 \
```

Here, we've used 8 processes for operations that don't use GPUs, namely to compute the speaking rate. If GPUs were present in the environnement, `num_workers_per_gpu` precises the number of processes per GPUs for the operations that can be computed on GPUs - namely pitch, SNR and reverberation estimation.

You can learn more about the arguments you can pass to `main.py` by passing:

```sh
python main.py --help
```

In addition to the command line used as an example, we've used the `repo_id` argument to push the dataset to the hub, resulting in [this dataset](https://huggingface.co/datasets/ylacombe/example_process_dataset).


TODO

### Map continuous annotations to key-words

TODO

### Generate natural language descriptions

TODO

### Perform audio separation


## License 
TODO: check license


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
