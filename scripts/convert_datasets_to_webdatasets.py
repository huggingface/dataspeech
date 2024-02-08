import webdataset as wds
import sys
from datasets import load_dataset, Audio, Value
import os
from pathlib import Path

# TODO: float doesn't really work -> the example change values in the lower decimal numbers

# dataset = load_dataset("collabora/whisperspeech", streaming=True)

dataset = load_dataset("ylacombe/english_dialects", "irish_male")

# dataset.cast_column("audio", Audio(sampling_rate=16000))

max_size = 1e8 # 1e9
max_count = 1000 # 100_000 

split = list(dataset.keys())[0]
dataset[split] = dataset[split].add_column("flot", [34.2]*len(dataset[split]))

int_columns = []
str_columns = []
audio_columns = []
float_columns = []

for key, value in dataset[split].features.items():
    if value.dtype == "string":
        str_columns.append(key)
    elif "int" in value.dtype:
        int_columns.append(key)
    elif value.dtype == "dict":
        audio_columns.append(key)
    elif "float" in value.dtype:
        float_columns.append(key)

for col in audio_columns:
    dataset = dataset.cast_column(col, Audio(decode=False))

base = "artefacts/irish_webdatasets/"
pattern = "irish"

pattern = os.path.join(base, f"{pattern}-{split}-%06d.tar")

Path(base).mkdir(parents=True, exist_ok=True)

with wds.ShardWriter(pattern, maxsize=int(max_size), maxcount=int(max_count)) as sink:
    for i, sample in enumerate(dataset[split]):

        key ="sample%08d" % i

        write_to = {"__key__": key}
        for col in int_columns + float_columns:
            write_to[f"{col}"] = str(sample[col]).encode("ascii") # bytes(sample[col]) #.cls # encode int
        for col in str_columns:
            write_to[f"{col}"] = sample[col] # bytes(sample[col], 'utf-8') #.txt
        for col in audio_columns:
            write_to[f"{col}"] = sample[col]["bytes"] #.wav  
            
        # Write the sample to the sharded tar archives.
        sink.write(write_to)

test_dataset = load_dataset("./artefacts/irish_webdatasets/")

for col in int_columns:
    test_dataset = test_dataset.cast_column(col, Value(dtype="int32"))
for col in float_columns:
    test_dataset = test_dataset.cast_column(col, Value(dtype="float"))
for col in audio_columns:
    test_dataset = test_dataset.cast_column(col, Audio())

print("ok")