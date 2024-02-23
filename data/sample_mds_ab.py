import json
import random
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import dask.array as da
import time
from streaming import MDSWriter

def target_name(file_name):
    return os.path.splitext(file_name.split("redpajama/")[1])[0] + ".npy"

def make_dir_if_not_ex(path):
    if not os.path.exists(path):
        print("Make target folder:", path)
        os.makedirs(path)

random_seed = 42
tokenized_folder = "redpajama" # fill in the path to the tokenized data

output_folder = sys.argv[1]

overwrite = False

train_eval_length_filter = int(sys.argv[2])
num_decoder_tokens = int(sys.argv[3])
prev_chunk_size = train_eval_length_filter - num_decoder_tokens

eval_target = int(sys.argv[4]) # The number of blocks to sample for evaluation
train_target = int(sys.argv[5]) # The number of blocks to sample for training

folders = {
    "arxiv": 0.5,
    "book": 0.5,
}
with open("all_jsonl.txt") as f:
    files = f.readlines()

target_folders = list(folders.keys()) 

folder_to_files = {f: [] for f in target_folders}
for line in files:
    tname = target_name(line.strip())
    for split in folders:
        if f"{split}/" in tname and split in target_folders:
            folder_to_files[split].append(os.path.join(tokenized_folder, tname))

def filter_length(data, split):
    # put each data point into chunks of specified lengths (different for train/eavl and retrieval)
    n = data[0].compute() if isinstance(data, da.Array) else data[0]
    lengths = data[1:n+1].compute() if isinstance(data, da.Array) else data[1:n+1]
    data = data[n+1:]
    chunks = []
    start = 0
    threshold = train_eval_length_filter
    for i in tqdm(range(n), desc=f"filtering {split} data", leave=False):
        length = lengths[i]
        end = start + length

        if length >= threshold:
            d = data[start:end]
            # we want to keep the prev tokens instead of the next tokens
            m = (length - prev_chunk_size) // num_decoder_tokens
            for j in range(m):
                e = prev_chunk_size + (j+1)*num_decoder_tokens
                assert e <= length
                x = d[j*num_decoder_tokens:e]
                assert x.size == num_decoder_tokens + prev_chunk_size
                chunks.append(x)

        start = end

    return chunks

# split files into eval, train
sampled_files = defaultdict(dict)
folder_to_data = defaultdict(dict)
ts = time.time()
for folder, files in folder_to_files.items():
    print(f"processing folder {folder}")
    files = [f for f in files if os.path.exists(f)]
    assert len(files) > 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    da.random.RandomState(random_seed)

    def split(x):
        n = max(len(x) // 100, 1)
        e = x[:n]
        t = x[n:len(x)//2]
        r = x[len(x)//2:]
        return e, t, r

    def handle_data(data):
        n = data[0]
        lengths = data[1:n+1]
        data = data[n+1:]
        sizes = n // 100
        # shuffle first
        buffer_ids = []
        buffer_lengths = []
        cm = np.cumsum(lengths)
        for i in range(100):
            start = cm[i*sizes - 1] if i > 0 else 0
            end = cm[min((i+1)*sizes, n)-1]
            buffer_ids.append(data[start:end])
            buffer_lengths.append(lengths[i*sizes:min((i+1)*sizes, n)])
        random_idx = list(range(100))
        random.shuffle(random_idx)
        buffer_ids = [buffer_ids[i] for i in random_idx]
        buffer_lengths = [buffer_lengths[i] for i in random_idx]

        # then divide in to eval, train, and retrieval
        ev = [da.concatenate([[len(buffer_lengths[0])], buffer_lengths[0], buffer_ids[0]])]
        tr = [da.concatenate([[len(buffer_lengths[i])], buffer_lengths[i], buffer_ids[i]]) for i in range(1, 50)]
        re = [da.concatenate([[len(buffer_lengths[i])], buffer_lengths[i], buffer_ids[i]]) for i in range(50, 100)]
        return ev, tr, re

    if len(files) == 1:
        sampled_files[folder]["eval"] = files
        sampled_files[folder]["train"] = files
        sampled_files[folder]["retrieval"] = files
        data = np.load(files[0], mmap_mode="r")
        # instead of shuffling the entire array, we split into 100 chunks and shuffle for a speedup
        e, t, r = handle_data(data)

        folder_to_data[folder]["eval"] = e
        folder_to_data[folder]["train"] = t
        folder_to_data[folder]["retrieval"] = r
    else:
        random.shuffle(files)
        e, t, r = split(files)
        sampled_files[folder]["eval"] = e
        sampled_files[folder]["train"] = t
        sampled_files[folder]["retrieval"] = r

        folder_to_data[folder]["eval"] = e
        folder_to_data[folder]["train"] = t
        folder_to_data[folder]["retrieval"] = r


print(f"preprocessing folders took {time.time()-ts} s, saving sampled files...")
# save the sampled files
with open(os.path.join(output_folder, "sampled_files.json"), "w") as f:
    json.dump(sampled_files, f, indent=4)

def sample_from_folder(files, folder, target, split):
    folder_target = int(target * folders[folder])
    print(f"total {folder_target} for split {split}")
    sample_per_file = max(1, folder_target // len(files)+1)

    make_dir_if_not_ex(os.path.join(output_folder, split))
    output_file = os.path.join(output_folder, split, folder)
    if os.path.exists(output_file) and not overwrite:
        print(f"already exists for {folder} {split}, skipping...")
        return

    columns = {
        "token_ids": "pkl",
        "prev_token_ids": "pkl",
        "domain": "str",
    }
    writer = MDSWriter(
        columns=columns,
        out=output_file,
        compression=None,
        size_limit="256mb",
    )
    count = 0
    cache = []

    for file in tqdm(files, desc=f"sampling {folder} {split}"):
        data = np.load(file, mmap_mode="r") if isinstance(file, str) else file
        data = filter_length(data, split)

        np.random.seed(random_seed+1)
        if len(data) < sample_per_file:
            print(f"warning {file}, {split} has less data than expected: {len(data)} < {sample_per_file}")
        indices = np.random.permutation(np.arange(len(data)))
        rest = indices[sample_per_file:]
        indices = indices[:sample_per_file]

        cache.extend([data[i] for i in rest])

        for index in tqdm(indices, leave=False, desc=f"writing {folder} {split}"):
            sample = data[index]
            d = sample.compute() if isinstance(sample, da.Array) else sample
            writer.write(
                {
                    "token_ids": d[-num_decoder_tokens:],
                    "prev_token_ids": d[:-num_decoder_tokens],
                    "domain": folder,
                }
            )
            count += 1
            if count >= folder_target:
                break

        if count >= folder_target:
            break

    if count < folder_target:
        np.random.seed(random_seed+1)
        indices = np.random.permutation(np.arange(len(cache)))
        for index in tqdm(indices, leave=False, desc=f"writing {folder} {split}"):
            sample = cache[index]
            writer.write(
                {
                    "token_ids": sample[-num_decoder_tokens:],
                    "prev_token_ids": sample[:-num_decoder_tokens],
                    "domain": folder,
                }
            )
            count += 1
            if count >= folder_target:
                break

    print(f"finished writing with {count} samples")
    writer.finish()

# Eval first
print("Sampling eval data...")
for folder in target_folders:
    print(f"domain: {folder}")
    selected = folder_to_data[folder]["eval"]
    sample_from_folder(selected, folder, eval_target, "eval")
print("done with eval")

# Train then
print("Sampling train data...")
for folder in target_folders:
    print(f"domain: {folder}")
    selected = folder_to_data[folder]["train"]
    sample_from_folder(selected, folder, train_target, "train")
print("Train done.")
