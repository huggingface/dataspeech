from g2p import make_g2p

transducer = make_g2p('eng', 'eng-ipa')

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