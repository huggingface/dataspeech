import re

from datasets import load_dataset
from deepmultilingualpunctuation import PunctuationModel
from multiprocess import set_start_method

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

import nltk
import spacy

model = PunctuationModel()

dataset_name = "ylacombe/mls-eng-tags"
output_dataset = "reach-vb/mls-eng-tags-spacy-v2"
process_split = "train"
proc = 16
device_id = 2

ds = load_dataset(dataset_name, split = process_split,  num_proc=proc)

spacy.require_gpu(gpu_id=device_id)

# Load the spaCy model
nlp = spacy.load('en_core_web_trf')

from spacy.util import compile_infix_regex

def custom_tokenizer(nlp):
    infixes = nlp.Defaults.infixes + ['\w+(?:-\w+)+']
    infix_regex = compile_infix_regex(infixes)
    return spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)

# Use the custom tokenizer
nlp.tokenizer = custom_tokenizer(nlp)

def true_case_spacy(text):
    # Process the text with the spaCy model
    doc = nlp(text)
    
    # Initialize an empty list to hold the processed sentences
    true_cased_sentences = []
    
    # Iterate through the sentences in the Doc object
    for sent in doc.sents:
        # Initialize an empty list to hold the processed tokens of the current sentence
        processed_tokens = []
        
        # Iterate through the tokens in the current sentence
        for i, token in enumerate(sent):
            # Capitalize the first word of the sentence and proper nouns
            if i == 0 or token.pos_ == 'PROPN':
                processed_tokens.append(token.text.capitalize())
            else:
                processed_tokens.append(token.text)
        
        # Join the processed tokens back into a string
        processed_sentence = ' '.join(processed_tokens)
        
        # Remove spaces between punctuations and the preceding word
        processed_sentence = re.sub(r'(\w) (\W)', r'\1\2', processed_sentence)
        
        # Add the processed sentence to the list of processed sentences
        true_cased_sentences.append(processed_sentence)
    
    # Join the processed sentences back into a single string
    true_cased_text = ' '.join(true_cased_sentences)
    
    return true_cased_text


def repunctuation_apply_simple(batch):
    
    repunct_sample = model.restore_punctuation(batch["text"])
    batch["repunct_text"] = true_case_spacy(repunct_sample)

    return batch

if __name__ == "__main__":
    set_start_method("spawn")
    repunct_ds = ds.map(repunctuation_apply_simple, batch_size=1, num_proc=proc)
    repunct_ds.push_to_hub(output_dataset, split = process_split)