import os

from dataclasses import dataclass
import glob
from typing import Any, Optional, Tuple, Dict

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import time
from transformers import DataCollatorForLanguageModeling

from streaming import Stream, StreamingDataset
from collections import defaultdict

RP_DOMAINS = ["arxiv", "book", "c4-rp", "cc", "github", "stackexchange", "wiki"]

class MLMDataset(Dataset):
    def __init__(self, path, domains = None, chunk_size=512):
        self.domains = RP_DOMAINS if domains is None else domains
        self.chunk_size = chunk_size

        print(f"loading from path {path} for domains {self.domains}")
        streams = [Stream(local=os.path.join(path, d)) for d in self.domains]
        self.dataset = StreamingDataset(streams=streams, allow_unsafe_types=True)

    def __len__(self):
        # MDS automatically divides total by world size to get len(), but we don't want to do that
        return self.dataset.epoch_size

    def __getitem__(self, idx):
        item = self.dataset[idx]
        ids = item["token_ids"][:self.chunk_size]
        item["input_ids"] = ids

        # create labels
        labels = np.copy(ids).astype(np.int32)
        item["labels"] = labels
        item["attention_mask"] = np.ones_like(ids)

        return item

    def reduce(self, size):
        self.input_ids = self.input_ids[:size]

@dataclass
class MLMDataCollator(DataCollatorForLanguageModeling):
    """
    The only change is that we disable 80-10-10, and we always replace with [MASK]
    """

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

@dataclass
class ReplugDataCollator:
    def __call__(self, batch):
        new_batch = {
            "input_ids": [],
            "labels": [],
            "context_scores": [],
            "attention_mask": []
        }

        # here we expect to see encoder input ids that contains the retrieved documents
        for item in batch:
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            ctx = torch.tensor(item["encoder_input_ids"], dtype=torch.long)
            n_ctx = ctx.size(0)

            # we have one copy of the input ids for each context
            ids = ids.view(1, -1).expand(n_ctx, -1)
            input_mask = torch.ones_like(ids)
            labels = torch.tensor(item["labels"]).long()
            labels = labels.view(1, -1).expand(n_ctx, -1)

            # -100 is ignored during loss calculations, which we want
            labels = torch.concat([torch.full_like(ctx, -100), labels], dim=1)

            # each context is prepended to one copy of the input, as well as their mask
            ids = torch.concat([ctx, ids], dim=1)
            if "encoder_attention_mask" not in item:
                context_mask = torch.ones_like(ctx, dtype=torch.long)
            else:
                context_mask = torch.tensor(item["encoder_attention_mask"], dtype=torch.long)
            attention_mask = torch.concatenate([context_mask, input_mask], dim=1)

            new_batch["input_ids"].append(ids)
            new_batch["labels"].append(labels)
            new_batch["attention_mask"].append(attention_mask)

            if "context_scores" in item:
                new_batch["context_scores"].append(torch.tensor(
                    item["context_scores"],
                    dtype=torch.float
                ))
            else:
                # uniform scores for the contexts if they don't have scores
                new_batch["context_scores"].append(torch.ones(n_ctx, dtype=torch.float))

        for key in new_batch:
            new_batch[key] = torch.stack(new_batch[key])

        return new_batch

class CombineStreamingDataset(Dataset):
    """
    This class allows us to combine multiple streaming datasets into one.
    The key motivation is to enable different modes of training.
    For example, we have the standard language modeling mode using the previous context as inputs (PrevDoc).
    We also have the retrieval mode where we use the retrieved documents as inputs (RetDoc). The retrieved documents may also have an associated retrieval score (this is necessary for RePlug).
    We will also incorporate the PMI score + the distillation into the training.

    Instead of writing everything to the same MDS dataset (which is not flexible), we write them to separate MDS datasets, and as long as the indexing is consistent between the datasets, we can combine them into one by calling getitem on each dataset separately first.

    We will also check if the remote starts with s3, if it does, then we use the remote argument. If it is already stored on local, then we will use the local argument.

    args:
        encoder_decoder_remote: the remote with the encoder-decoder input token ids; this is a required argument
        retrieval_remote: the remote with the retrieved documents, their and their neighbors' input token ids; this is an optional argument
        distill_remote: the remote with the distillation logits; this is an optional argument
        domains: the domains that we want to load.

        mask_prob: the probability of masking at context at all
        mask_seq_prob: the probability of masking the entire context when we do mask the seq

    """

    def __init__(
        self,
        encoder_decoder_remote,
        epoch_size=None,
        retrieval_remote=None,
        distill_remote=None,
        domains=None,
        num_context=8,
        context_size=256,
        chunk_size=256,
        loss_chunk_size=None,
        tokenizer=None,
        mask_prob=0.0,
        mask_seq_prob=0.0,
        load_strategy="best",
        retrieval_mode="no_neighbor",
    ):
        self.encoder_decoder_remote = encoder_decoder_remote
        self.retrieval_remote = retrieval_remote
        self.distill_remote = distill_remote
        self.epoch_size = epoch_size

        if domains is not None:
            domains = domains.split(";")
            domains = [d.split(",") for d in domains]
        self.domains = [RP_DOMAINS] if domains is None else domains
        self.num_context = num_context
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.loss_chunk_size = loss_chunk_size if loss_chunk_size is not None else chunk_size
        self.load_strategy = load_strategy

        self.tokenizer = tokenizer
        self.retrieval_mode = retrieval_mode
        self.mask_prob = mask_prob
        self.mask_seq_prob = mask_seq_prob

        self.load_streams()

    def get_streams(self, remote):
        paths = []
        remotes = remote.split(",")
        assert len(self.domains) == 1 or len(remotes) == len(self.domains), f"remote {remote} does not match the number of domains {self.domains}"
        for i, rem in enumerate(remotes):
            if len(self.domains) == 1:
                domains = self.domains[0]
            else:
                domains = self.domains[i]
            print(f"loading from remote: {rem} with domains {domains}")
            for d in domains:
                paths.append(f"{rem}/{d}")
        print(f"loading from paths {paths}")
        if remote.startswith("s3"):
            streams = [Stream(remote=path) for path in paths]
        else:
            streams = [Stream(local=path) for path in paths]
        return streams

    def load_streams(self):
        # we allow unsafe types because we save numpy arrays as pkl
        streams = self.get_streams(self.encoder_decoder_remote)
        self.encoder_decoder_dataset = StreamingDataset(streams=streams, epoch_size=self.epoch_size, allow_unsafe_types=True)

        self.distill_dataset = None
        if self.distill_remote is not None:
            self.distill_dataset = StreamingDataset(
                streams=self.get_streams(self.distill_remote), 
                epoch_size=self.epoch_size, 
                allow_unsafe_types=True
            )
            assert len(self.encoder_decoder_dataset) == len(self.distill_dataset), f"encoder-decoder dataset has length {len(self.encoder_decoder_dataset)} but distill dataset has length {len(self.distill_dataset)}"

        self.retrieval_dataset = None
        if self.retrieval_remote is not None:
            self.retrieval_dataset = StreamingDataset(
                streams=self.get_streams(self.retrieval_remote),
                epoch_size=self.epoch_size,
                allow_unsafe_types=True,
            )
            assert len(self.encoder_decoder_dataset) == len(self.retrieval_dataset) or self.retrieval_mode == "ignore", f"encoder-decoder dataset has length {len(self.encoder_decoder_dataset)} but retrieval dataset has length {len(self.retrieval_dataset)}"

    def get_item(self, sample_id):
        encoder_decoder_item = self.encoder_decoder_dataset[sample_id]

        # reshape the decoder input ids
        if self.chunk_size > encoder_decoder_item["token_ids"].shape[0]:
            assert self.num_context == 0 or self.load_strategy == "dummy" or self.load_strategy == "duplicate", "chunk size is greater than the total number of tokens in the document"
            # we might want to use some of the encoder inputs for decoder for some of the baselines with num_context = 0
            encoder_decoder_item["input_ids"] = np.concatenate([encoder_decoder_item["prev_token_ids"], encoder_decoder_item["token_ids"]])[-self.chunk_size:]
        else:
            encoder_decoder_item["input_ids"] = encoder_decoder_item["token_ids"][:self.chunk_size]

        if self.num_context > 0:
            if self.load_strategy == "dummy":
                encoder_decoder_item["encoder_input_ids"] = np.full((self.num_context, self.context_size), self.tokenizer.eos_token_id)
            elif self.load_strategy == "duplicate":
                total = self.context_size * self.num_context
                encoder_decoder_item["encoder_input_ids"] = encoder_decoder_item["input_ids"][:total].reshape(self.num_context, self.context_size)
            else:
                # need to reshape the encoder input ids to be (num_context, context_size)
                if self.retrieval_dataset is None or self.retrieval_mode == "ignore":
                    total = self.context_size * self.num_context
                    encoder_decoder_item["encoder_input_ids"] = encoder_decoder_item["prev_token_ids"][-total:].reshape(self.num_context, self.context_size)
                    if "context_scores" in encoder_decoder_item:
                        # assume that the scores match up with the preset context_size
                        if encoder_decoder_item["context_scores"].size != len(encoder_decoder_item["prev_token_ids"]) // self.context_size:
                            print("Warning: context scores size does not match up with the context size, this could be a problem for the RePlug models!!")

                        encoder_decoder_item["context_scores"] = encoder_decoder_item["context_scores"][-self.num_context:]
                        encoder_decoder_item["encoder_attention_mask"] = np.ones_like(encoder_decoder_item["encoder_input_ids"])

                else:
                    retrieval_item = self.retrieval_dataset[sample_id]
                    encoder_decoder_item["context_scores"] = retrieval_item["retrieval_scores"][:self.num_context]

                    passage_ids = retrieval_item["retrieved_token_ids"][:self.num_context]
                    neighbor_ids = retrieval_item["retrieved_neighbor_token_ids"][:self.num_context]
                    # we also need to handle the truncation and padding here since both ids are not guaranteed to be a fixed length
                    # alternatively we could use the tokenizer's padding function, but that would require us to rename the keys
                    def pad_and_truncate(ids, size):
                        # assume ids is batched
                        # return the ids and the mask
                        out_ids = []
                        mask = []
                        for id in ids:
                            if len(id) > size:
                                out_ids.append(id[:size])
                                mask.append(np.ones(size))
                            else:
                                out_ids.append(np.pad(
                                    id,
                                    (0, size - len(id)),
                                    mode="constant",
                                    constant_values=self.tokenizer.pad_token_id
                                ))
                                mask.append(np.pad(
                                    np.ones(len(id)),
                                    (0, size - len(id)),
                                    mode="constant",
                                    constant_values=0
                                ))
                        return np.array(out_ids), np.array(mask)

                    if self.retrieval_mode == "joint":
                        # we probably shouldn't insert the eos token here, since the two are guaranteed to be from the same doc
                        ids = [np.concatenate([pid, nid]) for pid, nid in zip(passage_ids, neighbor_ids)]
                        # we use 2x here because we have two things
                        ids, mask = pad_and_truncate(ids, 2*self.context_size)
                        encoder_decoder_item["encoder_input_ids"] = ids
                        encoder_decoder_item["encoder_attention_mask"] = mask

                    elif self.retrieval_mode == "separate":
                        id1, mask1 = pad_and_truncate(passage_ids, self.context_size)
                        id2, mask2 = pad_and_truncate(neighbor_ids, self.context_size)
                        encoder_decoder_item["encoder_input_ids"] = np.stack([id1, id2])
                        encoder_decoder_item["encoder_attention_mask"] = np.stack([mask1, mask2])

                    else:
                        # don't include neighbor
                        ids, mask = pad_and_truncate(passage_ids, self.context_size)
                        encoder_decoder_item["encoder_input_ids"] = ids
                        encoder_decoder_item["encoder_attention_mask"] = mask

        if self.distill_dataset is not None:
            distill_item = self.distill_dataset[sample_id]
            encoder_decoder_item["distill_prob"] = distill_item["target_prob"]
            encoder_decoder_item["distill_index"] = distill_item["target_index"]

        if self.mask_prob > 0:
            if "encoder_attention_mask" not in encoder_decoder_item:
                encoder_decoder_item["encoder_attention_mask"] = np.ones_like(encoder_decoder_item["encoder_input_ids"])

            # we sample a float between 0 and 1 for each context, and if it is less than mask_prob*mask_seq_prob, then we mask the entire context
            # if it is just less than mask_prob but greater than mask_prob*mask_seq_prob, then we randomly sample the number of tokens to mask
            # if it is greater than mask_prob, then we don't mask the context at all
            masks = np.random.uniform(size=self.num_context)
            for i, mask in enumerate(masks):
                if mask < self.mask_prob:
                    if mask < self.mask_prob * self.mask_seq_prob:
                        # mask the entire context
                        encoder_decoder_item["encoder_attention_mask"][i] = 0
                    else:
                        # randomly sample the number of tokens to mask
                        n_mask = np.random.randint(1, self.context_size)
                        encoder_decoder_item["encoder_attention_mask"][i][-n_mask:] = 0

        if "put_in_decoder" in self.load_strategy and self.num_context > 0:
            if "encoder_attention_mask" in encoder_decoder_item:
                encoder_decoder_item["attention_mask"] = np.concatenate([encoder_decoder_item.pop("encoder_attention_mask").reshape(-1), np.ones_like(encoder_decoder_item["input_ids"])])
            encoder_decoder_item["input_ids"] = np.concatenate([encoder_decoder_item.pop("encoder_input_ids").reshape(-1), encoder_decoder_item["input_ids"]])

        labels = np.copy(encoder_decoder_item["input_ids"]).astype(np.int32)
        if self.loss_chunk_size < self.chunk_size:
            labels[:-self.loss_chunk_size] = -100
        encoder_decoder_item["labels"] = labels

        return encoder_decoder_item

    def state_dict(self, num_samples: int, from_beginning: bool):
        return self.encoder_decoder_dataset.state_dict(num_samples, from_beginning)

    def load_state_dict(self, obj: Dict[str, Any]):
        self.encoder_decoder_dataset.load_state_dict(obj)
        if self.retrieval_dataset is not None:
            self.retrieval_dataset.load_state_dict(obj)
        if self.pmi_dataset is not None:
            self.pmi_dataset.load_state_dict(obj)
        if self.distill_dataset is not None:
            self.distill_dataset.load_state_dict(obj)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def __len__(self):
        # MDS automatically divides total by world size to get len(), but we don't want to do that
        return self.encoder_decoder_dataset.epoch_size
        # return len(self.encoder_decoder_dataset)

@dataclass
class ContextDataCollator:
    def __call__(self, batch):
        new_batch = defaultdict(list)

        for item in batch:
            new_batch["input_ids"].append(torch.tensor(item["input_ids"], dtype=torch.long))
            labels = torch.tensor(item["labels"], dtype=torch.long)
            new_batch["labels"].append(labels)
            if "encoder_input_ids" in item:
                new_batch["encoder_input_ids"].append(torch.tensor(item["encoder_input_ids"], dtype=torch.long))

            if "encoder_attention_mask" in item:
                new_batch["encoder_attention_mask"].append(torch.tensor(item["encoder_attention_mask"], dtype=torch.long))

            if "distill_prob" in item:
                new_batch["distill_prob"].append(torch.tensor(item["distill_prob"], dtype=torch.float32))
                new_batch["distill_index"].append(torch.tensor(item["distill_index"], dtype=torch.long))

        for key in new_batch:
            new_batch[key] = torch.stack(new_batch[key])
            if key == "encoder_input_ids" and len(new_batch[key].shape) == 4:
                # each item maybe have two encoder input, and we want to merge them in the second dimension
                # shape is (bsz, 2, num_context, context_size)
                new_batch[key] = new_batch[key].view(new_batch[key].size(0), -1, new_batch[key].size(-1))

        return dict(new_batch)
