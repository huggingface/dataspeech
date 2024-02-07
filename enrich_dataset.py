from datasets import load_dataset
from multiprocess import set_start_method
from scripts.enrich_pitch import pitch_apply
from scripts.enrich_snr_and_reverb import vad_apply
from scripts.enrich_speaking_rate import rate_apply 
import torch

# TODO: doesnt' work at all

from multiprocessing.managers import BaseManager, BaseProxy
class MyManager(BaseManager):
    pass

MyManager.register('pitch', pitch_apply)
MyManager.register('vad', vad_apply)
MyManager.register('rate', rate_apply)


def enrich(dataset):
    manager = MyManager()
    manager.start()


    updated_dataset = dataset.map(
        manager.rate,
        with_rank=False,
        num_proc=32 # TODO: num_workers
    )
    
    updated_dataset = updated_dataset.map(
        manager.pitch,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count()
    )
    updated_dataset = updated_dataset.map(
        manager.vad,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count()
    )
    return updated_dataset


    
if __name__ == "__main__":
    set_start_method("spawn")
    
    dataset = load_dataset("ylacombe/english_dialects", "irish_male")
    
    updated_dataset = enrich(dataset)
    updated_dataset.save_to_disk("artefacts/english_dialects_irish")
    updated_dataset.push_to_hub("ylacombe/english_dialects_with_tags", "irish_male")
    
    print("ok")
    
