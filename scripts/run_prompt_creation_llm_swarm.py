import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import logging

import math
import numpy as np
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
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
    num_instances: int = field(
        default=1,
        metadata={"help": "Number of TGI instances."},
    )
    per_instance_max_parallel_requests: int = field(
        default=500,
        metadata={"help": "Maximum number of parallel requests per instance."},
    )
    checkpoint_interval: Optional[int] = field(
        default=1000,
        metadata={
            "help": "Interval for streaming chunks of generation."
        },
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
    max_retries: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of retries per sample."},
    )
    retry_delay_in_s: Optional[float] = field(
        default=5.0,
        metadata={"help": "Time to wait between successive retries in seconds."},
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
    save_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Save the generated prompts every save_steps."},
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": ("If a value is passed, will limit the total number of saved checkpoints")}
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError(
                "You must specify the `hub_dataset_id` when setting `--push_to_hub=True`"
            )

CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+).json$")


def save_checkpoint(output_dir, all_generated_ids, step):
    checkpoint_path = f"{CHECKPOINT_PREFIX}-{step}.json"
    output_path = os.path.join(output_dir, checkpoint_path)
    with open(output_path, "w") as file:
        json.dump(all_generated_ids, file)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "r") as file:
        all_generated_ids = json.load(file)
    return all_generated_ids


def sorted_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        os.remove(checkpoint)


def get_last_checkpoint(folder) -> Tuple[List, int]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return [], 0
    last_checkpoint = os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+).json"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    # load corresponding generated ids
    all_generated_ids = load_checkpoint(last_checkpoint)
    return all_generated_ids, cur_step


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
        instances=model_args.num_instances,
        inference_engine="tgi",
        slurm_template_path="./tgi_h100.template.slurm",
        load_balancer_template_path="./nginx.template.conf",
        model=model_args.model_name_or_path,
        revision=model_args.model_revision,
        per_instance_max_parallel_requests=model_args.per_instance_max_parallel_requests,
        debug_endpoint=model_args.debug_endpoint,
    )
) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
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
        attempt = 0
        while attempt < model_args.max_retries:
            try:
                async with semaphore:
                    return await client.text_generation(
                        prompt=sample_prompt,
                        max_new_tokens=model_args.max_new_tokens,
                        temperature=model_args.temperature,
                        do_sample=model_args.do_sample,
                    )
            except Exception as e:
                attempt += 1
                if attempt < data_args.max_retries:
                    print(
                        f"Request failed due to {e}, retrying in {data_args.retry_delay_in_s} seconds... (Attempt {attempt}/{data_args.max_retries})"
                    )
                    await asyncio.sleep(data_args.retry_delay_in_s)
                else:
                    raise ValueError(
                        f"Max retries reached. Failed with error: {e}."
                    )

    async def main():
        # 2. Setup logging
        logger.setLevel(logging.INFO)
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
            total_samples = len(raw_datasets[split])
            total_inference_steps = math.ceil(total_samples / model_args.checkpoint_interval)

            split_output_dir = os.path.join(data_args.output_dir, split)
            progress_bar = tqdm(range(total_inference_steps), desc=f"{split}", position=0)

            all_generated_ids, inference_step = get_last_checkpoint(split_output_dir)
            if inference_step > 0:
                logger.info(f"Resuming {split} from step {inference_step}")
                progress_bar.update(inference_step)

            while inference_step < total_inference_steps:
                start_index = inference_step * model_args.checkpoint_interval
                end_index = min((inference_step + 1) * model_args.checkpoint_interval, total_samples)
                inference_chunk = raw_datasets[split].select(range(start_index, end_index))
                results = await asyncio.gather(
                    *(process_text(sample) for sample in inference_chunk)
                )
                inference_step += 1
                progress_bar.update(1)
                all_generated_ids.extend(results)

                if (inference_step % data_args.save_steps == 0) or (inference_step == total_inference_steps):
                    logger.info(f"Saving generations of step {inference_step}")
                    save_checkpoint(split_output_dir, all_generated_ids, inference_step)
                    rotate_checkpoints(data_args.save_total_limit, output_dir=split_output_dir)

            raw_datasets[split] = raw_datasets[split].add_column(
                "text_description", all_generated_ids
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
