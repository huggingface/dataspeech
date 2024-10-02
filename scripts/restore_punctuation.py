import argparse
from multiprocessing import set_start_method
from datasets import load_dataset
from deepmultilingualpunctuation import PunctuationModel
import spacy
from typing import Dict, Callable, List
import re

nlp_models: Dict[str, spacy.language.Language] = {}

def load_spacy_model(lang_code: str) -> spacy.language.Language:
    """Load and return the appropriate spaCy model for the given language code."""
    if lang_code not in nlp_models:
        model_name = {
            'ca': 'ca_core_news_sm',
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
            'it': 'it_core_news_sm',
            'pl': 'pl_core_news_sm',
            'nl': 'nl_core_news_sm',
            'pt': 'pt_core_news_sm',
        }.get(lang_code)
        
        if model_name is None:
            raise ValueError(f"Unsupported language code: {lang_code}")
        
        nlp_models[lang_code] = spacy.load(model_name)
    
    return nlp_models[lang_code]

def get_capitalization_function(lang_code: str) -> Callable[[spacy.tokens.Token], str]:
    """Return the appropriate capitalization function for the given language."""
    
    def default_capitalization(token: spacy.tokens.Token) -> str:
        if token.is_sent_start or token.pos_ in ('PROPN', 'NNP', 'NNPS'):
            return token.text.capitalize()
        return token.text.lower()
    
    def german_capitalization(token: spacy.tokens.Token) -> str:
        if token.is_sent_start or token.pos_ in ('PROPN', 'NOUN'):
            return token.text.capitalize()
        return token.text.lower()
    
    if lang_code == 'de':
        return german_capitalization
    else:
        return default_capitalization

def true_case(text: str, lang_code: str) -> str:
    """
    Perform true casing on the input text for the specified language.
    
    :param text: Input text to be true cased
    :param lang_code: Two-letter language code (e.g., 'en' for English)
    :return: True cased text
    """
    nlp = load_spacy_model(lang_code)
    capitalization_func = get_capitalization_function(lang_code)
    
    doc = nlp(text)
    true_cased_tokens = [capitalization_func(token) for token in doc]
    
    # Join tokens, ensuring no space before punctuation
    true_cased_text = ""
    for i, token in enumerate(doc):
        if i > 0 and not token.is_punct:
            true_cased_text += " "
        true_cased_text += true_cased_tokens[i]
    
    return true_cased_text

def remove_quotes(text: str) -> str:
    """Remove single and double quotes from the input text."""
    return re.sub(r"['\"]", "", text)

def apply_processing(examples, punctuation_model, text_column, description_column, lang_code, punctuation_only, truecase_only):
    result = {}
    
    # Process text column
    if text_column:
        if punctuation_only:
            processed_texts = [punctuation_model.restore_punctuation(text) for text in examples[text_column]]
        elif truecase_only:
            processed_texts = [true_case(text, lang_code) for text in examples[text_column]]
        else:
            restored_texts = [punctuation_model.restore_punctuation(text) for text in examples[text_column]]
            processed_texts = [true_case(text, lang_code) for text in restored_texts]
        
        result[f"original_{text_column}"] = examples[text_column]
        result[text_column] = processed_texts
    
    # Process description column
    if description_column:
        if punctuation_only:
            processed_descriptions = [remove_quotes(punctuation_model.restore_punctuation(text)) for text in examples[description_column]]
        elif truecase_only:
            processed_descriptions = [remove_quotes(true_case(text, lang_code)) for text in examples[description_column]]
        else:
            restored_descriptions = [punctuation_model.restore_punctuation(text) for text in examples[description_column]]
            processed_descriptions = [remove_quotes(true_case(text, lang_code)) for text in restored_descriptions]
        
        result[f"original_{description_column}"] = examples[description_column]
        result[description_column] = processed_descriptions
    
    return result

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--text_column", default="text", type=str, help="Name of the column containing the main text to be processed.")
    parser.add_argument("--description_column", default="text_description", type=str, help="Name of the column containing the description text to be processed.")
    parser.add_argument("--language", default=None, type=str, help="Language of the dataset. If not specified, uses the default multilingual model.")
    parser.add_argument("--batch_size", default=32, type=int, help="This parameter specifies how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--punctuation_only", action="store_true", help="If set, only perform punctuation restoration.")
    parser.add_argument("--truecase_only", action="store_true", help="If set, only perform true casing.")
    parser.add_argument("--process_text_only", action="store_true", help="If set, only process the text column.")
    parser.add_argument("--process_description_only", action="store_true", help="If set, only process the description column.")

    args = parser.parse_args()

    if args.punctuation_only and args.truecase_only:
        raise ValueError("Cannot set both --punctuation_only and --truecase_only. Choose one or neither.")

    if args.process_text_only and args.process_description_only:
        raise ValueError("Cannot set both --process_text_only and --process_description_only. Choose one or neither.")

    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

    language_to_code = {
        "catalan": "ca",
        "english": "en",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "polish": "pl",
        "dutch": "nl",
        "portuguese": "pt",
    }

    supported_languages = set(language_to_code.keys())
    if args.language and args.language.lower() not in supported_languages:
        raise ValueError(f"Language {args.language} is not supported. Please choose from: {', '.join(supported_languages)}")
    
    lang_code = language_to_code[args.language.lower()] if args.language else 'en'

    if lang_code == "ca":    
        punctuation_model = PunctuationModel(model="softcatala/fullstop-catalan-punctuation-prediction")
    elif lang_code in {"en", "it", "fr", "de", "nl"}:
        punctuation_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilingual-base")
    else:
        punctuation_model = PunctuationModel(model="kredor/punctuate-all")

    # Determine which columns to process
    text_column = args.text_column if not args.process_description_only else None
    description_column = args.description_column if not args.process_text_only else None

    processed_dataset = dataset.map(
        apply_processing,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs={
            "punctuation_model": punctuation_model,
            "text_column": text_column,
            "description_column": description_column,
            "lang_code": lang_code,
            "punctuation_only": args.punctuation_only,
            "truecase_only": args.truecase_only
        },
        desc="Processing text"
    )

    if args.output_dir:
        print("Saving to disk...")
        processed_dataset.save_to_disk(args.output_dir)
    
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            processed_dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            processed_dataset.push_to_hub(args.repo_id)

    print("Processing completed for the following columns:", ", ".join(filter(None, [text_column, description_column])))
    if args.punctuation_only:
        print("Operation: Punctuation restoration")
    elif args.truecase_only:
        print("Operation: True casing")
    else:
        print("Operation: Punctuation restoration and true casing")
    if description_column:
        print("Note: Single and double quotes were removed from the description column")