# Data

In this directory, we pre-process [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1) for training. 

To get started, you want to first download RedPajama using the instruction from the [original repo](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep).
You should obtain one directory for each of the seven domains in RedPajama, where each directory contains jsonl files.
You'll want to put all jsonl files into a .txt file for future steps.
To do this, you can run 
```
python get_all_jsonl.py {PATH TO YOUR DATA}
```
This will read all the files from the data directory and save all the jsonl file relative paths in all_jsonl.txt.

## Tokenization
To make filtering easy, we tokenize everything first:
```
bash run_tokenize.sh
```
This will convert each .jsonl file into a .npy file containing the tokens of the documents.
To keep the original document boundaries, we save the tokens in the format of:
```
[n] [len(d1), ..., len(dn)] [d1] ... [dn]
```
where `n` is the number of documents in the file and is the first index of the numpy array, the next `n` indices are the length of each document, followed by the tokens of each document.
This allows us to access the length of each document and filter them with ease. 

If you are using a slurm system, then you can distribute the tokenization process using slurm array jobs. 
The code supports tokenizing in shards for this purpose, refer to the code for more details. 

Note: We use the Transformers tokenizer in this version, but we would highly recommend using the SentencePiece tokenizer. 
We found it to be much faster than the Transformers implementation in practice, particularly for shorter sequences.
On the other hand, the Transformers TokenizerFast is faster on longer sequences (helpful when preprocessing the books domain for example). 
You can check it out [here](https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py).

You can run this for a quick sanity check:
```
python sanity_check.py
```

## Sampling
Then, we sample the data for training, testing, and retrieval. 
We encode them as MDS, the format used by [Mosaic's streaming package](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).
This gives us a lot of flexbility of modifying the data during training easily while loading them in a memory-efficient manner. 

To sample the Arxiv and Books domain and filter by a total length (RP filtered):
```
python sample_mds_ab.py {OUTPUT_PATH} {TOTAL LENGTH} {DECODER LENGTH} {EVAL SAMPLES} {TRAIN SAMPLES}
python sample_mds_ab.py 8k_ab 8192 4096 1000 5000000
python sample_mds_ab.py 32k_ab 32768 4096 1000 0
python sample_mds_ab.py 128_ab 131072 4096 1000 0
```
Due to the way we structure the sampling, the files/documents used in the evaluation set should be always the same across different run with consistent random seeds.

Similar, we sample the RP concatenate with
```
python sample_mds_concat.py {OUTPUT_PATH} {ENCODER LENGTH} {DECODER LENGTH} {EVAL SAMPLES} {TRAIN SAMPLES} {RETRIEVAL SAMPLES} {SHARD ID}

for $shard in $(seq 0 99); do
    python sample_mds_concat.py 8k_concat 4096 4096 35000 2500000 2e6 $shard
done
```
By default, we split retrieval into 100 shards. If you are not interested in preprocessing the retrieval split, you can set `retrieval_shard = 0` in `sample_mds_concat.py`.
The retrieval files will be saved as jsonl files across 100 shards. If you are interested in building and using the retrieval corpus, please refer to the `retrieval` directory.

## CEPED

To train CEPED, we require annotating the training data with logits of a base model.
```
python calculate_logits.py \
    --input_dir $DATA_PATH \
    --model_dir $MODEL \
    --domain $DOMAIN \
    --num_decoder_tokens $N_DEC \
    --num_encoder_tokens $N_ENC \
    --topk $TOPK \
    --shard_id $SHARD_ID \
    --retrieval_input_dir $RET_DATA \
    --num_shards $N_SHARD --merge
```

To do this with LLaMA-2-Chat-7B as the base model on the 8k_ab_train dataset, arxiv domain:
```
for shard in $(seq 0 99); do
    python calculate_logits.py \
        --input_dir 8k_ab/train \
        --model_dir meta-llama/Llama-2-7b-chat-hf \
        --domain arxiv \
        --num_decoder_tokens 2048 \
        --num_encoder_tokens 2048 \
        --shard_id 0 \
        --num_shards 100
done
```
If you decide to use sharding across different jobs, you can merge them with:
```
python calculate_logits.py \
    --input_dir 8k_ab/train \
    --model_dir meta-llama/Llama-2-7b-chat-hf \
    --domain arxiv \
    --num_decoder_tokens 2048 \
    --num_encoder_tokens 2048 \
    --shard_id 0 \
    --num_shards 100 --merge
```

Note that this step requires a lot of storage -- annotating 2.5M sequences (~10B tokens) takes about 1.5T.
