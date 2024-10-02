from datasets import load_dataset
import argparse
import re

def capitalize_first_letter(text):
    return '. '.join(sentence.capitalize() for sentence in text.split('. '))

def capitalize_words_remove_quotes(text):
    # Remove quotes and capitalize each word
    return ' '.join(word.capitalize() for word in re.findall(r'\w+', text))

def apply_recasing(examples, text_column, description_column):
    recased_texts = [capitalize_first_letter(text) for text in examples[text_column]]
    recased_descriptions = [capitalize_words_remove_quotes(desc) for desc in examples[description_column]]
    return {
        f"original_{text_column}": examples[text_column],
        text_column: recased_texts,
        f"original_{description_column}": examples[description_column],
        description_column: recased_descriptions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--text_column", default="text", type=str, help="Name of the column containing the text to be recased.")
    parser.add_argument("--description_column", default="text_description", type=str, help="Name of the column containing the description to be recased and cleaned.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for processing.")
    parser.add_argument("--num_proc", default=1, type=int, help="Number of processes to use.")

    args = parser.parse_args()

    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.num_proc)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.num_proc)

    recased_dataset = dataset.map(
        apply_recasing,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs={"text_column": args.text_column, "description_column": args.description_column},
        desc="Applying recasing"
    )

    if args.output_dir:
        print("Saving to disk...")
        recased_dataset.save_to_disk(args.output_dir)
    
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            recased_dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            recased_dataset.push_to_hub(args.repo_id)

    print("Recasing completed.")