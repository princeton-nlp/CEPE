import os

import pickle
import json
import numpy as np
import glob
from tokenize_files import shard_size

domains = ["arxiv", "book", "c4-rp", "cc", "github", "stackexchange", "wiki"]
def check():
    with open("all_jsonl.txt") as f:
        files = f.readlines()
        ids = set()
        missing_files = []
        for i, file in enumerate(files):
            file = file.strip()
            target_name = os.path.splitext(file.split("redpajama/")[1])[0] + ".npy"
            if not os.path.exists(target_name) and "stackexchange" not in target_name:
                missing_files.append(target_name)
                ids.add(i // shard_size)
        #assert len(missing_files) == len(ids)
        print(f"num missing files: {len(missing_files)}")
        iid = ','.join([str(i) for i in sorted(list(ids))])
        print(f"remaining array ids to run: {iid}")
        print()

def stats():
    total_chunks = 0
    counts = {}
    chunk_size = 4096
    for domain in domains:
        files = glob.glob(f"{domain}/**/*.npy", recursive=True)
        print(f"found {len(files)} files for {domain}")
        num_chunks = 0
        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            for d in data:
                num_chunks += len(d) // chunk_size

        tokens = num_chunks * chunk_size
        counts[domain] = tokens
        print(f"found {num_chunks} chunks for {domain}, totaling to {tokens/1e9:.02f}B tokens")
        print()
        total_chunks += num_chunks

    print(f"total found {total_chunks} chunks, totaling to {total_chunks*4096/1e9:.02f}B tokens\n")
    for k, v in counts.items():
        print(f"{k},{v}")

if __name__ == "__main__":
    check()
    stats()