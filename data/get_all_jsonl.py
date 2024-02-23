# from https://github.com/princeton-nlp/LLM-Shearing/blob/main/llmshearing/data/get_all_jsonl.py

import os
import sys

root_dir = sys.argv[1]

with open("all_jsonl.txt", "w") as fp:
    for root, ds, fs in os.walk(root_dir):
        for f in fs:
            full_path = os.path.join(root, f)
            relative_path = full_path[len(root_dir):].lstrip(os.sep) 
            if relative_path.endswith(".jsonl"):
                print(relative_path)
                fp.write(relative_path + "\n") 