

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
- Annotate an audio speech dataset using `main.py` to get the following continuous variables:
    - Speaking rate `(nb_phonemes / utterance_length)`
    - Speech-to-noise ratio
    - Reverberation
    - Pitch estimation
- Map the previous annotations categorical to discrete keywords bins using `scripts/metadata_to_text.py`
- Create natural language descriptions from a set of keywords using `scripts/run_prompt_creation.py`

Moreover, additional scripts cover other use-cases such as:
- perform audio separation using [demucs](TODO) in a multi-GPU settings
- add gender information to [MLS](TODO) and [LibriTTS-R](TODO).
- more to come...

> [!TIP]
> Scripts from this library can also be used as a starting point for applying other models to datasets from the [datasets library](TODO) in a multi-machine configuration.
> 
> For example, `scripts/run_prompt_creation.py` can be adapted to perform large-scaled inference using other LLMs and prompts.

### Generates annotations

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
