import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

import logging
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
import asyncio
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from tqdm.asyncio import tqdm_asyncio


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "The name of the model to use (via the transformers library) for the prompt annotation."
        },
    )
    per_instance_max_parallel_requests: int = field(
        default=500,
        metadata={"help": "Maximum number of parallel requests per instance."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use sampling mode for generation"}
    )
    temperature: Optional[float] = field(
        default=0.6, metadata={"help": "Temperature for sampling-based generation"}
    )
    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "Maximum number of new tokens during generation"}
    )
    token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use an authentication token when loading/uploading from the Hugging Face Hub"
        },
    )
    debug_endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Endpoint to use for debugging (e.g. http://localhost:13120)."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split name of the dataset to use (via the datasets library)."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of samples for generation - use for debugging purposes."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory each time the script is run."
        },
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError(
                "You must specify the `hub_dataset_id` when setting `--push_to_hub=True`"
            )


# TODO(SG): add accent
EXPECTED_COLUMNS = {
    "gender",
    "pitch",
    "noise",
    "reverberation",
    "speech_monotony",
    "speaking_rate",
}

PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (e.g., male, female)
2. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
3. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
4. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
5. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)
6. The pitch of the speaker's voice (e.g., very low pitch, quite low pitch, slightly low pitch, moderate pitch, slightly high pitch, quite high pitch, very high pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.

For example, given the following keywords: 'female', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly', a valid description would be: 'a woman with a deep voice speaks slowly but has an animated delivery in an echoey room with some background noise'.

For the keywords: '[gender]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]', the corresponding description is:"
"""


# 1. Parse input arguments
parser = HfArgumentParser((ModelArguments, DataArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )
else:
    model_args, data_args = parser.parse_args_into_dataclasses()

with LLMSwarm(
    LLMSwarmConfig(
        instances=1,
        inference_engine="tgi",
        slurm_template_path="./tgi_h100.template.slurm",
        load_balancer_template_path="./nginx.template.conf",
        model=model_args.model_name_or_path,
        revision=model_args.model_revision,
        per_instance_max_parallel_requests=model_args.per_instance_max_parallel_requests,
        debug_endpoint=model_args.debug_endpoint,
    )
) as llm_swarm:
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )

    async def process_text(sample):
        sample_prompt = PROMPT
        for key in EXPECTED_COLUMNS:
            sample_prompt = sample_prompt.replace(f"[{key}]", sample[key])
        sample_prompt = [{"role": "user", "content": sample_prompt}]
        sample_prompt = tokenizer.apply_chat_template(sample_prompt, tokenize=False)
        return await client.text_generation(
            prompt=sample_prompt,
            max_new_tokens=model_args.max_new_tokens,
            temperature=model_args.temperature,
            do_sample=model_args.do_sample,
        )

    async def main():
        # 2. Setup logging
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if (
            data_args.overwrite_output_dir
            and os.path.exists(data_args.output_dir)
            and os.path.isdir(data_args.output_dir)
        ):
            logger.info("Cleaning output dir from previous run...")
            shutil.rmtree(data_args.output_dir)

        # 3. Load annotated dataset
        logger.info("*** Load annotated dataset ***")
        if data_args.dataset_split_name is not None:
            raw_datasets = DatasetDict()
            data_splits = data_args.dataset_split_name.split("+")
            # load on a split-wise basis
            for split in data_splits:
                raw_datasets[split] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    num_proc=data_args.preprocessing_num_workers,
                )
        else:
            # load all splits for annotation
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )

        raw_datasets_features = set(
            raw_datasets[next(iter(raw_datasets))].features.keys()
        )

        if data_args.max_eval_samples is not None:
            for split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].select(
                    range(data_args.max_eval_samples)
                )

        if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
            missing_columns = EXPECTED_COLUMNS - raw_datasets_features
            raise ValueError(
                f"Missing columns {missing_columns} from the dataset features. Got dataset features {raw_datasets_features}"
            )

        for split in raw_datasets:
            results = await tqdm_asyncio.gather(
                *(process_text(sample) for sample in raw_datasets[split])
            )
            raw_datasets[split] = raw_datasets[split].add_column(
                "text_description", results
            )

        raw_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            raw_datasets.push_to_hub(
                data_args.hub_dataset_id,
                config_name=(
                    data_args.dataset_config_name
                    if data_args.dataset_config_name is not None
                    else "default"
                ),
                token=model_args.token,
            )

    asyncio.run(main())
