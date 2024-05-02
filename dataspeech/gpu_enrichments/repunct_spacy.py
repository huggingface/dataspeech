import re

from datasets import load_dataset
from deepmultilingualpunctuation import PunctuationModel
from multiprocess import set_start_method

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

import nltk
import spacy

# from rpunct import RestorePuncts

# rpunct = RestorePuncts()

model = PunctuationModel()


ds = load_dataset("ylacombe/mls-eng-tags", split = "train",  num_proc=16)

def truecasing_by_pos(input_text):
    
    # break input text to sentences
    sent_texts = sent_tokenize(input_text)
    
    full_text = ""

    for sent_text in sent_texts:
        # tokenize the text into words
        words = word_tokenize(sent_text)

        # apply POS-tagging on words
        tagged_words = pos_tag([word.lower() for word in words])
    
        # apply capitalization based on POS tags
        capitalized_words = [w.capitalize() if t in ["NNP","NNPS"] else w for (w,t) in tagged_words]
    
        # capitalize first word in sentence
        capitalized_words[0] = capitalized_words[0].capitalize()
    
        # join capitalized words
        text_truecase = " ".join(capitalized_words)

        full_text += text_truecase.strip()

    return full_text.strip()

def true_case(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Process each sentence
    true_cased_sentences = []
    for sentence in sentences:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)

        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)

        # Capitalize the first word of the sentence and NNP and NNPS tags
        for i, (word, tag) in enumerate(tagged):
            if i == 0 or tag in ('NNP', 'NNPS'):
                tagged[i] = (word.capitalize(), tag)

        # Join tokens back into a string, preserving punctuation
        true_cased_sentence = ' '.join(word for word, tag in tagged)

        # Remove spaces between punctuations and the preceding word
        true_cased_sentence = re.sub(r'(\w) (\W)', r'\1\2', true_cased_sentence)

        true_cased_sentences.append(true_cased_sentence)

    # Join the processed sentences back into a single string
    true_cased_text = ' '.join(true_cased_sentences)

    return true_cased_text

spacy.require_gpu(gpu_id=2)

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
    repunct_ds = ds.map(repunctuation_apply_simple, batch_size=1, num_proc=14)
    repunct_ds.push_to_hub("reach-vb/mls-eng-tags-spacy-v2", split = "train")