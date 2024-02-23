# Long-Context Language Modeling with Parallel Encodings

<!-- <p align="center"><img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/moose.png?raw=true" alt="ALCE" width="15%"><br>*: ALCE is pronounced as /elk/ as ALCE is the Latin word for elk (Europe) or moose (North America). -->
<!-- </p> -->

This repository contains the code and data for paper [Long-Context Language Modeling Parallel Encodings](). 
In this work, we propose CAPE -- **C**ross-**Attention** to **P**arallel **E**ncodings -- a flexible framework for extending the context window of language models. 
This repository includes the code for preprocessing the data, training CAPE, and evaluating all baselines.

<img src="https://github.com/princeton-nlp/CAPE/blob/main/assets/overview.png?raw=true" alt="CAPE" width="100%">


## Quick Start

Want to test out CAPE? Try out our models using Huggingface!
First, set up the environment by installing the [requirements](#requirements).
Then, simply copy and paste the `modeling_llama_flash.py` file into your working directory, and then you can run it with:
```python
import torch
from transformers import LlamaTokenizer
from modeling_llama_flash import LlamaForCausalContextLM

device = "cuda"
model_name = "hyen/CAPED-LLaMA-2-Chat-7B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalContextLM.from_pretrained(
  model_name,
  use_flash_attention_2="flash_attention_2", 
  torch_dtype=torch.bfloat16,
  device_map="auto",
).eval()

contexts = tokenizer([
    "My friends and I enjoy eating out at restaurants together. However, we also enjoy cooking and making food as a group as well."
    "Many of my friends like to play soccer and volleyball. We also enjoy watching movies and going to museums and galleries.",
], return_tensors="pt", padding=True)

inputs = tokenizer("Question: what are three ideas for a social with a large groups of friends in New York City.\nAnswer:", return_tensors="pt")

output = model.generate(
  input_ids=inputs.input_ids.to(device), 
  attention_mask=inputs.attention_mask.to(device), 
  encoder_input_ids=contexts.input_ids.unsqueeze(0).to(device),
  encoder_attention_mask=contexts.attention_mask.unsqueeze(0).to(device), 
  max_new_tokens=200,
)
print(tokenizer.batch_decode(output))
```

This prints out:
```
Answer: Here are three ideas for a social with a large group of friends in New York City:
1. Host a rooftop party: Find a rooftop bar or restaurant with a great view of the city. Invite a large group of friends and enjoy drinks, appetizers, and music together.
2. Go on a group scavenger hunt: Create a list of items or challenges that your friends must complete around the city. This could include taking a photo with a street performer, buying a drink from a specific bar, or finding a landmark. The group that completes the most challenges wins.
3. Take a group cooking class: Find a cooking school or culinary institute in New York City that offers classes for large groups. Choose a theme or type of cuisine to focus on, such as Italian or Asian. Then, work together to prepare a meal as a group.
```

## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Training](#reproducing-baselines)
  - [Language Modeling Evaluation](#language-modeling-evaluation)
  - [Downstream Evaluation](#downstream-evaluation)
  - [Bug or Questions](#bug-or-questions)
  - [Citation](#citation)


## Requirements

Please install the latest versions of PyTorch (`torch`) by following the [official installation instructions](https://pytorch.org/get-started/locally/).
You can install the rest of the requirements with `pip install --r requirements.txt`. 
We also recommend checking out the installation instruction for [flash-attention](https://github.com/Dao-AILab/flash-attention). 
We tested the code with `python==3.9.12`, `torch==2.1.1`, `accelerate==0.24.0`, `transformers==4.34.1`, and CUDA version 12.3.

## Data

To obtain the training and evaluation data, please refer to the `./data` directory.

### Retrieval

It can be expensive to build and index the retrieval corpus at a large scale.
We provide the test set used in the Retrieval-augmented Language Modeling section on google drive, which you can download from this [link](https://drive.google.com/drive/folders/1QrnhOpnpGP2aPRyq0AUmjX5eZIZFLb5T?usp=sharing).

If you are interested in doing more with retrieval, you can reproduce our retrieval corpus by following the steps described in the `./data` directory and the `./retrieval` directory. 
We model our retrieval code after the Contriever code [repository](https://github.com/facebookresearch/contriever).


### CAPED

## Code Structure

* `train.py`: train CAPE
* `eval_lm.py`: evaluate CAPE and other baselines on language modeling
* `eval_downstream.py`: evaluate CAPE and other baselines on downstream tasks
* `configs/`: folder that contains all config files to reproduce baselines

## Training

## Language Modeling Evaluation

## Downstream Evaluation



## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Howard (`hyen@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please cite our paper if you use CAPE in your work:

<!-- ```bibtex
TODO: 
``` -->
