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

## Usage


## How does it work




Its 
TODO: check license



TODO:
- [] Current scripts only work on a single machine
- [] Benchmark for best dataset format
- [] Script conversion to webdatasets
- [] Silence removal at the beginning and at the end of the audio
- [] Speaking rate (nb_phonemes / utterance_length) for multilingual ?
- [] Make it compatible with streaming -> much more interesting ?



Don't forget to describe that you create another dataset

Don't forget to add examples script

Don't forget to add utility to merge datasets
            # TODO: probably should clean phonemes before using it for speaking_rate 


Don't forget to add accent classifier + LLM annotation

TODO: clean metadata
TODO add tools to visualize

TODO: shoutout to datasets and other libraries