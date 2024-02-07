
mkdir artefacts/
wget https://huggingface.co/ylacombe/brouhaha-best/resolve/main/best.ckpt?download=true -O artefacts/best.ckpt


TODO:
- [] Benchmark for best dataset format
- [] Script conversion to webdatasets
- [] Silence removal at the beginning and at the end of the audio
- [] Speaking rate (nb_phonemes / utterance_length) for multilingual ?