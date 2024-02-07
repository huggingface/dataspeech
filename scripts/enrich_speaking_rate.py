from g2p import make_g2p
from datasets import load_dataset
from multiprocess import set_start_method



transducer = make_g2p('eng', 'eng-ipa')


# TODO: make compatible with streaming
# TODO: make compatible with other naming
def rate_apply(batch, rank=None):
    if isinstance(batch["audio"], list):  
        speaking_rates = []
        phonemes_list = []
        for text, audio in zip(batch["text"], batch["audio"]):
            # TODO: change "text"
            # TODO: probably should clean phonemes before using it for speaking_rate 
            phonemes = transducer(text).output_string
            
            sample_rate = audio["sampling_rate"]
            audio_length = len(audio["array"].squeeze()) / sample_rate
            
            speaking_rate = len(phonemes) / audio_length

            
            speaking_rates.append(speaking_rate)
            phonemes_list.append(phonemes)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = transducer(batch["text"]).output_string
            
        sample_rate = batch["audio"]["sampling_rate"]
        audio_length = len(batch["audio"]["array"].squeeze()) / sample_rate
        
        speaking_rate = len(phonemes) / audio_length
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch
    
if __name__ == "__main__":
    set_start_method("spawn")
    dataset = load_dataset("ylacombe/english_dialects", "irish_male")
    
    # sampling_rate = next(iter(dataset))["audio"]["sampling_rate"]    
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    updated_dataset = dataset.map(
        rate_apply,
        batched=True,
        batch_size=16,
        with_rank=False,
        #num_proc=24
    )
    
    updated_dataset.save_to_disk("artefacts/english_dialects_irish")
    # updated_dataset.push_to_hub("ylacombe/english_dialects", "irish_male")
    
    print("ok")
    
