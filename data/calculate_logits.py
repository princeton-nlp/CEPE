import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from streaming import StreamingDataset, MDSWriter, Stream
from tqdm import tqdm

from transformers import LlamaConfig, LlamaForCausalLM


def merge(args, output_dir):
    output_dir = os.path.join(output_dir, args.domain)
    print(f"Merging {output_dir}")
    writer = MDSWriter(out=output_dir, columns={
        "target_prob": "pkl",
        "target_index": "pkl",
    })

    dataset = StreamingDataset(streams=[Stream(local=output_dir+f"-{shard_id}") for shard_id in range(args.num_shards)], allow_unsafe_types=True)
    for i in tqdm(range(len(dataset)), ):
        writer.write(dataset[i])

    writer.finish()
    print(f"Done merging {args.domain} with {len(dataset)} data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/10m_sample_4k_chunks_ab/eval")
    parser.add_argument("--retrieval_input_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="/scratch/gpfs/hyen/models/Llama-2-7b-chat-hf")
    parser.add_argument("--num_decoder_tokens", type=int, default=2048)
    parser.add_argument("--num_encoder_tokens", type=int, default=2048)
    parser.add_argument("--num_ret_passages", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--retrieval_k", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--domain", type=str, default="arxiv")
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()
    input_dir = args.input_dir
    model_dir = args.model_dir
    if args.retrieval_input_dir is not None:
        output_dir = args.retrieval_input_dir + f"-ret_k{args.retrieval_k}-dec{args.num_decoder_tokens}-" + os.path.basename(model_dir)

    else:
        output_dir = args.input_dir + f"-logits-enc{args.num_encoder_tokens}-dec{args.num_decoder_tokens}-" + os.path.basename(model_dir)

    if args.merge:
        merge(args, output_dir)
        exit()

    config = LlamaConfig.from_pretrained(model_dir)
    config._flash_attn_2_enabled = True
    model = LlamaForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=torch.bfloat16)
    print("model:", model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    num_decoder_tokens = args.num_decoder_tokens
    num_encoder_tokens = args.num_encoder_tokens
    topk = args.topk

    ret_k = args.retrieval_k

    with torch.inference_mode():
        domain = args.domain
        dataset = StreamingDataset(
            local=os.path.join(input_dir, domain),
            shuffle=False,
            num_canonical_nodes=1,
            allow_unsafe_types=True,
        )
        output_file = os.path.join(output_dir, domain + f"-{args.shard_id}")

        ret_data = None
        if args.retrieval_input_dir is not None:
            ret_data = StreamingDataset(
                local=os.path.join(args.retrieval_input_dir, domain),
                shuffle=False,
                num_canonical_nodes=1,
                allow_unsafe_types=True,
            )

        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists")
            exit()

        writer = MDSWriter(out=output_file, columns={
            "target_prob": "pkl",
            "target_index": "pkl",
            # "token_ids": "pkl",
            # "prev_token_ids": "pkl",
        })

        shard_size = len(dataset) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(dataset)

        for i in tqdm(range(start_idx, end_idx), desc=f"Calculating logits for {domain}"):
            item = dataset[i]

            if ret_data is None:
                encoder_input_ids = torch.tensor(item["prev_token_ids"][-num_encoder_tokens:], device=device)
            else:
                # we ignore the neighbors here...
                encoder_input_ids = torch.tensor(np.concatenate(ret_data[i]["retrieved_token_ids"][:ret_k]), device=device)

            decoder_input_ids = torch.tensor(item["token_ids"][:num_decoder_tokens], device=device)

            input_ids = torch.cat((encoder_input_ids, decoder_input_ids)).unsqueeze(0)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits.squeeze(0)[-num_decoder_tokens:]

            prob = F.softmax(logits, dim=-1)

            # save logits
            target_prob, target_indices = torch.sort(prob, descending=True, dim=-1)

            target_prob_topk = torch.cat([target_prob[:, :topk], target_prob[:, topk:].sum(dim=1, keepdim=True)], dim=1).to(torch.float16).cpu().numpy()
            target_indices_topk = target_indices[:, :topk].cpu().numpy()

            writer.write({
                "target_prob": target_prob_topk.astype(np.float16),
                "target_index": target_indices_topk.astype(np.int32),
                # "token_ids": decoder_input_ids.cpu().numpy(),
                # "prev_token_ids": encoder_input_ids.cpu().numpy(),
            })

        writer.finish()


if __name__ == "__main__":
    main()
