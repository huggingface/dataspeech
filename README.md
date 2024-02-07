
mkdir artefacts/
wget https://huggingface.co/ylacombe/brouhaha-best/resolve/main/best.ckpt?download=true -O artefacts/best.ckpt


TODO:
- [] Parallelism for VAD, SNR, C50
- [] Benchmark for parallelism
- [] Benchmark for best dataset format
- [] Silence removal at the beginning and at the end of the audio
- [] Speaking rate (nb_phonemes / utterance_length) 