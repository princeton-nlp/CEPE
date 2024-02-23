import os
import json
import time

from tqdm import tqdm
import numpy as np
from transformers import LlamaTokenizerFast
from llama_tokenizer import Tokenizer

shard_size = 1
overwrite = False

def process(file_name, tokenizer):
    target_name = os.path.splitext(file_name.split("redpajama/")[1])[0] + ".npy"
    if os.path.exists(target_name) and not overwrite:
        print(f"Target file {target_name} already exists, skipping")
        return
    target_folder = os.path.dirname(target_name)
    print(f"File path: {file_name}\ntarget folder: {target_folder}\ntarget name: {target_name}")
    if not os.path.exists(target_folder):
        print("Make target folder:", target_folder)
        os.makedirs(target_folder, exist_ok=True)

    all_ids = []
    s_time = time.time()
    # batch encode with fast is supposed to get speed up
    with open(file_name) as f:
        #all_text = [json.loads(line)["text"] for line in f]
        buffer = []
        count = 0
        for i, line in enumerate(tqdm(f)):
            try:
                item = json.loads(line.strip())
            except Exception as e:
                print(f"warning!!!! failed to load one of the lines as json.")
                continue
            text = item["text"]
            buffer.append(text)
            count += 1
            # tbh im not sure if a buffer is that necessary, but this is to avoid OOM in case of large files with too many lines
            if len(buffer) >= 10:
                max_len = max([len(s) for s in buffer])
                if max_len >= 5e6:
                    print(f"warning, found a file with {max_len} chars, this will likely take long")
                if isinstance(tokenizer, Tokenizer):
                    all_tokens = [tokenizer.encode(text, bos=True, eos=True) for text in buffer]
                else:
                    all_tokens = tokenizer(buffer, add_special_tokens=False).input_ids
                    all_tokens = [[tokenizer.bos_token_id] + t + [tokenizer.eos_token_id] for t in all_tokens]

                all_ids += all_tokens
                buffer = []

        print(f"finished reading file, processing buffer with {len(buffer)} items")
        if len(buffer) > 0:
            if type(tokenizer) == Tokenizer:
                all_tokens = [tokenizer.encode(text, bos=True, eos=True) for text in buffer]
            else:
                all_tokens = tokenizer(buffer, add_special_tokens=False).input_ids
                all_tokens = [[tokenizer.bos_token_id] + t + [tokenizer.eos_token_id] for t in all_tokens]
            all_ids += all_tokens
        assert len(all_ids) == count, f"expected {count} ids, but only found {len(all_ids)}"

        print(f"total num documents: {count}")

    print(f"Finished tokenization in {time.time() - s_time:.2f} seconds")
    lengths = [len(id) for id in all_ids]
    # format: n = # docs (1 int), cumulative sum of the lengths (n ints), the ids
    data = np.concatenate([[len(lengths)]] + [lengths] + all_ids)
    np.save(target_name, data)
    print(f"Done, saved to {target_name}, total num chunks: {len(all_ids)}")

def main():
    index_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    with open("all_jsonl.txt") as f:
        files = f.readlines()
        assert index_id*shard_size < len(files)
        file_names = files[index_id*shard_size:(index_id+1)*shard_size]

    # process shard_size files at a time
    tokenizer = LlamaTokenizerFast.from_pretrained("https://huggingface.co/meta-llama/Llama-2-7b-hf")
    # alternatively, use the Tokenizer class
    # tokenizer = Tokenizer("path/to/tokenizer.model")

    for file_name in file_names:
        process(file_name.strip(), tokenizer)
    print(f"all done!")

if __name__ == "__main__":
    main()
