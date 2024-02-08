
mkdir artefacts/
wget https://huggingface.co/ylacombe/brouhaha-best/resolve/main/best.ckpt?download=true -O artefacts/best.ckpt


TODO:
- [] Current scripts only work on a single machine
- [] Benchmark for best dataset format
- [] Script conversion to webdatasets
- [] Silence removal at the beginning and at the end of the audio
- [] Speaking rate (nb_phonemes / utterance_length) for multilingual ?
- [] Make it compatible with streaming -> much more interesting