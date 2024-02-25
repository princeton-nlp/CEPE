import os

import argparse
from collections import defaultdict
import copy
import json
import random
import math
import re
import yaml

from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizerFast, LlamaConfig, set_seed

from modeling_llama_flash import LlamaForCausalContextLM, LlamaForReplugCausalLM
from utils import (
    normalize_answer,
    drqa_exact_match_score,
    drqa_metric_max_over_ground_truths,
    get_max_memory,
    f1_score,
    substring_exact_match_score,
    nll_acc,
    nll_acc_norm,
    nll_acc_calibrated,
    nll_acc_calibrated_norm,
)
from dataset_utils import (
    load_data,
    load_qa_templates,
    load_hf_dataset,
    add_mmlu_options,
    add_boolq_options,
    preprocess_alce,
    DATASET_TO_TASK,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def sample_demos(data, num):
    # sample the demos using a balanced sampling strategy from each class

    n_classes = len(data)
    n_sets = math.ceil(num / n_classes)
    new_data = [[] for _ in range(n_sets)]

    for classes in data:
        idx = random.sample(range(len(classes)), n_sets % len(classes))
        while len(idx) < n_sets:
            # if we want more sets than the number of demos in this class, we just sample with replacement
            idx += random.sample(range(len(classes)), len(classes))

        for i, id in enumerate(idx):
            new_data[i].append(classes[id])
    for d in new_data:
        random.shuffle(d)
    new_data = [item for sublist in new_data for item in sublist][:num]

    return new_data

def preprocess_demos(demos, balanced_sampling=False):
    # preprocess the demos by grouping them by answer if we are doing balanced sampling
    # else we put all demos into one group
    if not balanced_sampling:
        return [list(demos)]
    by_answer = defaultdict(list)
    for item in demos:
        by_answer[item["answer"]].append(item)
    return [v for k, v in by_answer.items()]

def calibrate_nll(data, device, domain_prompt, tokenizer, model):
    """
    Calibrate the log likelihood of the continuations without context
    """

    scores = []
    prompt_tokens = tokenizer([domain_prompt], return_tensors="pt").input_ids
    prompt_length = prompt_tokens.size(1)
    if prompt_length == 0:
        prompt_tokens = torch.tensor([[tokenizer.bos_token_id]])
        prompt_length = 1
    with torch.inference_mode():
        for option in data["options"]:
            inputs = tokenizer([domain_prompt + option], return_tensors="pt", add_special_tokens=False)
            input_ids = torch.cat([prompt_tokens, inputs.input_ids], dim=1).to(torch.int).to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits.detach().cpu()[0, prompt_length-1:-1]
            prob = F.softmax(logits, dim=-1)
            score = prob[torch.arange(inputs.input_ids.size(1)), inputs.input_ids[0]]
            score = torch.log(score).sum()
            scores.append(score)
    return torch.stack(scores)

# TODO: this can probably be moved to the utils file
class TestItem:
    """
    Test item
    """
    def __init__(self, args, all_data, test_item, dataset):
        """
        This class represents a test item, which consists of a list of demos and a test item.
        self.demos = [{"docs": ["Title + text of doc"], "text": "the QA pair"}], length = args.shot
        self.test_documents = ["Title + text of doc"], length = args.n_test_doc
        self.test_document_scores = [score], length = args.n_test_doc (for replug)
        self.test_text = "the question"
        self.answer = "the answer"
        self.continuations = ["the continuation"], length = 4 (for multiple choice questions) else 0
        """
        demos = all_data["train"]
        if dataset == "popqa":
            # https://github.com/AlexTMallen/adaptive-retrieval/blob/beec2683e08ca4061ca9e0841d2d129528f534a1/run_model.py#L237
            demos = [[item for item in demos[0] if item["question"] != test_item["question"]]]

        demos = sample_demos(demos, args.shot)
        self.task = DATASET_TO_TASK[dataset]

        # for each demo, we keep a list of documents and the actual text/prompt
        self.demos = []
        for demo in demos:
            d = {"docs": []}
            if args.n_demo_doc > 0:
                ctxs = demo["ctxs"] if "ctxs" in demo else demo["docs"]
                for i in range(min(args.n_demo_doc, len(ctxs))):
                    doc_temp = all_data["document_template"]
                    ctx = ctxs[i]
                    d["docs"].append(doc_temp.format(**ctx, idx=i+1))
            if dataset == "nq" or dataset == "triviaqa" or dataset == "popqa":
                # there are multiple answers so we just take the first
                if isinstance(demo["answer"], list):
                    demo["answer"] = " "+demo["answer"][0]

            instruction = "" if not args.use_instruction else all_data["instruction"]
            if args.use_instruction and "alce" in dataset:
                instruction = instruction.split("\n\n")[0] + f" Given {args.n_demo_doc} documents, the citations that you can use are " + "".join([f"[{i+1}]" for i in range(args.n_demo_doc)]) + ".\n\n"

            text = all_data["template"].format(**demo, instruction=instruction)
            d["text"] = text
            d["question"] = demo["question"] if "question" in demo else None
            self.demos.append(d)

        # for the test item, we keep a list of documents
        self.test_documents = []
        self.test_documents_scores = []

        if args.n_test_doc > 0:
            ctxs = test_item["ctxs"] if "ctxs" in test_item else test_item["docs"]
            for i in range(min(args.n_test_doc, len(ctxs))):
                doc_temp = all_data["document_template"]
                ctx = ctxs[i]
                self.test_documents.append(doc_temp.format(**ctx, idx=i+1))
                # contriever code saves score as strings
                self.test_documents_scores.append(float(ctx["score"]) if "score" in ctx else 0.0)

        self.question = test_item["question"] if "question" in test_item else None
        self.answer = test_item["answer"]
        test_item["answer"] = ""
        instruction = "" if not args.use_instruction else all_data["instruction"]
        if args.use_instruction and "alce" in dataset:
            instruction = instruction.split("\n\n")[0] + f" Given {len(self.test_documents)} documents, the citations that you can use are " + "".join([f"[{i+1}]" for i in range(len(self.test_documents))]) + ".\n\n"
        self.test_text = all_data["template"].format(**test_item, instruction=instruction)
        test_item["answer"] = self.answer

        if self.task == "loglikelihood":
            self.answer_idx = test_item["options"].index(self.answer)
            assert self.answer_idx >= 0

        # the continuations for this test item
        self.continuations = [] if "options" not in test_item else test_item["options"]
        self.truncate_seperator = all_data["truncate_seperator"]

        if args.model_class == "cepe" and "concat" in args.context_strategy:
            # in this case, we are just moving the context to encoder, maybe we should find a better prompt for this
            #self.truncate_seperator = "... [The rest of the context has been moved to the start of the input]\n\n"
            self.truncate_seperator = "\n\n"

    def format_documents(self, docs):
        # format the documents for the context input
        return "\n\n".join(docs)

    def format_demo(self, demo, num_doc):
        # format the demos for the context input
        return self.format_documents(demo["docs"][:num_doc]) + ("\n\n" if num_doc > 0 else "") + demo["text"]

    def tokenize_continuations(self, tokenizer):
        max_continuation_length = 0
        continuations = []

        for continuation in self.continuations:
            # then we get the input ids for the decoder
            continuation_inputs = tokenizer([continuation], return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
            continuation_length = continuation_inputs.input_ids.size(1)

            continuations.append({
                "input_ids": continuation_inputs.input_ids,
                "attention_mask": continuation_inputs.attention_mask,
                "continuation_length": continuation_length,
            })
            max_continuation_length = max(max_continuation_length, continuation_length)

        return continuations, max_continuation_length

    def format_decoder_inputs(self, args, demos, num_demo_doc, test_docs, test_text, input_max_length, tokenizer):
        if "passage_at_front" in args.context_strategy:
            # add all the passages first and then the texts
            text = "\n\n".join([self.format_documents(demo["docs"][:num_demo_doc]) for demo in demos if num_demo_doc > 0])
            if len(test_docs) > 0:
                text += "\n\n" + self.format_documents(test_docs)
                text += "\n\n" if len(test_text) > 0 else ""
            query_start_index = len(text) # we would rather truncate the passages than the demos

            demos_text = "\n\n".join([demo["text"] for demo in demos]) + ("\n\n" if len(demos) > 0 else "")
            text += demos_text
            if query_start_index == 0:
                query_start_index = len(text)
                text += test_text
            else:
                # in case of truncation
                text += test_text
                test_text = demos_text + test_text

        else:
            text = "\n\n".join([self.format_demo(demo, num_demo_doc) for demo in demos])
            text += "\n\n" if len(text) > 0 else ""
            text += "\n\n".join(test_docs) + ("\n\n" if len(test_docs) > 0 else "")

            query_start_index = len(text)
            text += test_text

        tokenized_text = tokenizer([text], return_tensors="pt")
        if tokenized_text.input_ids.shape[1] <= input_max_length:
            return tokenized_text.input_ids, tokenized_text.attention_mask, ""

        # we need to truncate the text
        logger.info(f"Prompt length exceeds max input length: {tokenized_text.input_ids.shape[1]} > {input_max_length}, truncating...")

        test_text = self.truncate_seperator + test_text
        tokenized_query = tokenizer([test_text], return_tensors="pt", add_special_tokens=False)
        before_query = text[:query_start_index]
        tokenized_before_query = tokenizer([before_query], return_tensors="pt", return_offsets_mapping=True)

        n_context_tokens = input_max_length - tokenized_query.input_ids.size(1)
        offset_mapping = tokenized_before_query.offset_mapping[0]
        max_tok = 2**18
        if "concat" in args.context_strategy:
            matches = re.findall(r'concat\d+', args.context_strategy)
            max_tok = int(matches[0][6:]) if len(matches) > 0 else max_tok

        if "truncate_left" in args.context_strategy:
            input_ids = tokenized_before_query.input_ids[:, -n_context_tokens:]
            start_tok = max(-tokenized_before_query.input_ids.size(1), -n_context_tokens-max_tok+1)
            overflown_text = text[offset_mapping[start_tok][0]:offset_mapping[-n_context_tokens][1]]
        else:
            input_ids = tokenized_before_query.input_ids[:, :n_context_tokens]
            end_tok = min(tokenized_before_query.input_ids.size(1)-1, n_context_tokens+max_tok-1)
            overflown_text = text[offset_mapping[n_context_tokens][0]:offset_mapping[end_tok][1]]

        input_ids = torch.cat([input_ids, tokenized_query.input_ids], dim=1)

        return input_ids, torch.ones_like(input_ids), overflown_text

    def get_vanilla_inputs(self, args, tokenizer):
        # vanilla is just a simple concatenation of the prompt
        model_inputs = {}

        continuation_inputs, max_continuation_length = self.tokenize_continuations(tokenizer)
        model_inputs["continuation_inputs"] = continuation_inputs

        input_max_length = args.input_max_length - (args.generation_max_length if self.task == "generate" else max_continuation_length)

        prefix_input_ids, prefix_attn_mask, overflown_text = self.format_decoder_inputs(args, self.demos, args.n_demo_doc, self.test_documents, self.test_text, input_max_length, tokenizer)

        prefix_length = prefix_input_ids.size(1)
        model_inputs["prefix_length"] = prefix_length
        model_inputs["prefix_inputs"] = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attn_mask
        }

        return model_inputs

    def get_replug_inputs(self, args, tokenizer):
        # TODO: update all this
        constant_demos = self.demos[:args.n_shot_decoder]
        # we don't allow diff demos because they are not used in the replug paper, and we don't have the Contriever scores
        #diff_demos = self.demos[args.n_shot_decoder:]
        constant_docs = self.test_documents[:args.n_test_doc_decoder]
        diff_docs = self.test_documents[args.n_test_doc_decoder:args.n_test_doc]
        constant_prompt = "\n\n".join([self.format_demo(demo, args.n_demo_doc_decoder) for demo in constant_demos])
        constant_prompt += "\n\n" if len(constant_prompt) > 0 else ""
        constant_prompt += "\n\n".join(constant_docs) + ("\n\n" if len(constant_docs) > 0 else "")
        # diffs = [self.format_demo(demo, args.n_demo_doc_encoder) for demo in diff_demos] + diff_docs
        diffs = diff_docs

        # demo_prompt = "\n\n".join(["\n".join(d["docs"]) + "\n" + d["text"] for d in self.demos]) + "\n\n"
        # # for replug, we parallelize the test documents
        # prefix_prompts = [demo_prompt + test_doc + "\n" + self.test_text for test_doc in self.test_documents]
        prefix_prompts = [constant_prompt + diff + "\n\n" + self.test_text for diff in diffs]

        # n_context = len(self.test_documents)
        n_context = len(diffs)
        context_scores = torch.tensor(self.test_documents_scores[args.n_test_doc_decoder:args.n_test_doc]).unsqueeze(0)

        model_inputs = {}
        continuation_inputs, max_continuation_length = self.tokenize_continuations(tokenizer)
        model_inputs["continuation_inputs"] = continuation_inputs

        prefix_inputs = tokenizer(
            prefix_prompts,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length - max_continuation_length,
        )
        prefix_length = prefix_inputs.input_ids.size(1)
        model_inputs["prefix_length"] = prefix_length
        model_inputs["prefix_inputs"] = {
            "input_ids": prefix_inputs.input_ids,
            "attention_mask": prefix_inputs.attention_mask,
            "num_context": n_context,
            "context_scores": context_scores,
        }

        return model_inputs

    def get_context_inputs(self, args, tokenizer):
        """
        The context input can be the demos, the documents, or overflowing tokens from the encoder, or a combination of these
        """
        context_text = []
        decoder_demos = self.demos[:args.n_shot_decoder]
        encoder_demos = self.demos[args.n_shot_decoder:]
        decoder_docs = self.test_documents[:args.n_test_doc_decoder]
        encoder_docs = self.test_documents[args.n_test_doc_decoder:args.n_test_doc]

        # step 1: get the continuation inputs
        model_inputs = {}
        continuation_inputs, max_continuation_length = self.tokenize_continuations(tokenizer)
        model_inputs["continuation_inputs"] = continuation_inputs

        input_max_length = args.input_max_length - (args.generation_max_length if self.task == "generate" else max_continuation_length)

        # step 2: get the decoder inputs (called prefix here)
        prefix_input_ids, prefix_attn_mask, overflown_text = self.format_decoder_inputs(args, decoder_demos, args.n_demo_doc_decoder, decoder_docs, self.test_text, input_max_length, tokenizer)
        prefix_length = prefix_input_ids.size(1)

        # step 3: get the encoder inputs -- three possible things to put in the encoder
        # 1. the demos
        pre_extra = ""
        extra = ""
        if "include_all" in args.context_strategy:
            extra = "\n\n" + self.test_text
        elif "include_question" in args.context_strategy:
            extra = "\n\n" + f"Question: {self.question}"
        elif "include_query" in args.context_strategy:
            pre_extra = f"Search query: {self.question}\nSearch results:\n"

        demos_in_encoder = [self.format_demo(demo, args.encoder_demo_n_doc) for demo in encoder_demos]
        demos_in_encoder = [
            "\n\n".join(demos_in_encoder[i:i+args.n_shot_encoder]) + extra
            for i in range(0, len(demos_in_encoder), args.n_shot_encoder)
        ] if args.n_shot_encoder > 0 else []

        # 2. the documents for demo in the decoder
        decoder_demo_doc_in_encoder = []
        if args.n_demo_doc > args.n_demo_doc_decoder:
            for demo in decoder_demos:
                temp_docs = demo["docs"][args.n_demo_doc_decoder:args.n_demo_doc]

                extra_demo = ""
                pre_extra_demo = ""
                if "include_all" in args.context_strategy:
                    extra_demo = "\n\n" + demo["text"]
                elif "include_question" in args.context_strategy:
                    extra_demo = "\n\n" + f"Question: {demo['question']}"
                elif "include_query" in args.context_strategy:
                    pre_extra_demo = f"Search query: {demo['question']}\nSearch results:\n"

                decoder_demo_doc_in_encoder += [
                    pre_extra_demo + self.format_documents(temp_docs[i:i+args.n_demo_doc_encoder]) + extra_demo
                    for i in range(0, len(temp_docs), args.n_demo_doc_encoder)
                ]

        # 3. the test documents
        docs_in_encoder = [
            pre_extra + self.format_documents(encoder_docs[i:i+args.n_test_doc_encoder]) + extra
            for i in range(0, len(encoder_docs), args.n_test_doc_encoder)
        ] if args.n_test_doc_encoder > 0 else []

        context_text = demos_in_encoder + decoder_demo_doc_in_encoder + docs_in_encoder

        # we add the overflown_text if we are using concat
        if "concat" in args.context_strategy:
            context_text = ["\n\n".join(demos_in_encoder), "\n\n".join(decoder_demo_doc_in_encoder), "\n\n".join(docs_in_encoder), overflown_text]
            context_text = [text for text in context_text if len(text) > 0]

        # get the encoder input ids (called context inputs here)
        if len(context_text) > 0:
            # TODO: add a smart packing logic here, it's just concatenate that respects the boundary within max length
            if "smart" in args.context_strategy:
                context_inputs = tokenizer(context_text)
                new_contexts = []
                cur_text = []
                cur_length = 0
                for i, ids in enumerate(context_inputs.input_ids):
                    # we are joining the texts with two newlines --> 2 extra tokens
                    if cur_length == 0 or cur_length + len(ids) + 2 <= args.context_max_length:
                        cur_text.append(context_text[i])
                        cur_length += len(ids) + (2 if cur_length > 0 else 0)
                    else:
                        new_contexts.append("\n\n".join(cur_text))
                        cur_text = [context_text[i]]
                        cur_length = len(ids)
                if len(cur_text) > 0:
                    new_contexts.append("\n\n".join(cur_text))
                context_text = new_contexts

            context_inputs = tokenizer(
                context_text,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                max_length=args.context_max_length,
                return_overflowing_tokens="concat" in args.context_strategy,
                truncation=True,
                add_special_tokens=False,
            )
                   
            # unsqueeze bc we expect shape of bsz, n_context, seq_len
            encoder_input_ids= context_inputs.input_ids.unsqueeze(0)
            encoder_attention_mask= context_inputs.attention_mask.unsqueeze(0)

        else:
            # this is the rare case where we don't have any context
            # (e.g. zero-shot and the context can fit within the decoder)
            encoder_input_ids = None
            encoder_attention_mask = None

        # put everything into the model inputs
        model_inputs["prefix_length"] = prefix_length
        model_inputs["prefix_inputs"] = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attn_mask,
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
        }
        for inputs in continuation_inputs:
            inputs["encoder_input_ids"] = encoder_input_ids
            inputs["encoder_attention_mask"] = encoder_attention_mask

        return model_inputs


class TestItemDataset(Dataset):
    def __init__(self, args, all_data, dataset, tokenizer):
        self.args = args
        self.all_data = all_data
        self.dataset = dataset
        self.test_data = all_data["test"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        test_item = TestItem(self.args, self.all_data, self.test_data[idx], self.dataset)

        if self.args.model_class == "cepe":
            inputs = test_item.get_context_inputs(self.args, self.tokenizer)
        elif self.args.model_class == "replug":
            inputs = test_item.get_replug_inputs(self.args, self.tokenizer)
        elif self.args.model_class == "vanilla":
            inputs = test_item.get_vanilla_inputs(self.args, self.tokenizer)
        inputs["original_data"] = self.test_data[idx]
        inputs["test_item"] = test_item
        return inputs


def run_test(args, tokenizer, model, device, dataset, test_file, demo_file):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    assert dataset in DATASET_TO_TASK, f"dataset {dataset} not supported"
    logger.info(f"using dataset {dataset}")

    if args.output_dir is not None:
        output_path = os.path.join(args.output_dir,
            f"eval-{dataset}-" +
            os.path.splitext(os.path.basename(test_file))[0] +
            f"-{args.tag}-model_class{args.model_class}" +
            (f"-samples{args.max_test_samples}" if args.max_test_samples is not None else "")
            + f"-shot{args.shot}enc{args.n_shot_encoder}doc{args.encoder_demo_n_doc}dec{args.n_shot_decoder}-n_demo_doc{args.n_demo_doc}enc{args.n_demo_doc_encoder}dec{args.n_demo_doc_decoder}-n_test_doc{args.n_test_doc}enc{args.n_test_doc_encoder}dec{args.n_test_doc_decoder}-gen_len{args.generation_max_length}_{args.generation_min_length}input_len{args.input_max_length}context_len{args.context_max_length}-strat{args.context_strategy}-cali{args.calibrate_nll}empty{args.empty_domain_prompt}-inst{args.use_instruction}-samp{args.do_sample}t{args.temperature}p{args.top_p}-{args.seed}.json"
        )
        if os.path.exists(output_path) and not args.overwrite:
            logger.info(f"output path {output_path} already exists, use --overwrite if you want to run the test again")
            return output_path

    # load data
    if os.path.isfile(test_file) and os.path.isfile(demo_file):
        logger.info(f"loading test data from {test_file}")
        test_data = load_data(test_file)
        logger.info(f"loading demo data from {demo_file}")
        demos_data = load_data(demo_file)

        if "alce" in dataset:
            demos_data, test_data, templates = preprocess_alce(args, demos_data, test_data)
        else:
            # if local, we are probably using a QA dataset
            templates = load_qa_templates(dataset, args.include_title)

        # remove test questions from demos, replug does this so we follow it
        # however, popqa uses the same set of questions for demos and test
        if dataset != "popqa":
            if dataset == "mmlu":
                # for mmlu, we only have 5 demos from the dev set, so we cannot afford to filter any of them out
                # so we remove overlapping questions from the test set
                demo_questions = [normalize_answer(item["question"]) for item in demos_data]
                test_data = [item for item in test_data if normalize_answer(item["question"]) not in demo_questions]
            else:
                # for other datasets, we remove duplicates from the demo data
                test_questions = [normalize_answer(item["question"]) for item in test_data]
                demos_data = [item for item in demos_data if normalize_answer(item["question"]) not in test_questions]

        # shuffle before potentially cutting to prevent artifacts
        random.shuffle(test_data)
        if args.max_test_samples is not None:
            logger.info(f"using only {args.max_test_samples} test samples")
        test_data = test_data[:args.max_test_samples]

        all_data = {"train": demos_data, "test": test_data}
        all_data.update(templates)

    else:
        all_data = load_hf_dataset(dataset, demo_file, test_file)
        if args.max_test_samples is not None:
            all_data["test"].shuffle(seed=args.seed)
            max_samples = min(len(all_data["test"]), args.max_test_samples)
            all_data["test"] = all_data["test"].select(range(max_samples))

    # special cases...
    if dataset == "mmlu":
        all_data["train"] = add_mmlu_options(all_data["train"])
        all_data["test"] = add_mmlu_options(all_data["test"])
    elif dataset == "boolq":
        all_data["train"] = add_boolq_options(all_data["train"])
        all_data["test"] = add_boolq_options(all_data["test"])

    assert args.shot <= len(all_data["train"]), f"only have {len(all_data['train'])} possible demos, not enough for {args.shot} shots"
    logger.info(f"running evaluation with {len(all_data['test'])} test data and {len(all_data['train'])} possible demos.")
    all_data["train"] = preprocess_demos(all_data["train"], all_data["balanced_sampling"])

    if args.calibrate_nll and DATASET_TO_TASK[dataset] == "loglikelihood" and not all_data["recalibrate_every"] :
        logger.info("calibrating the log likelihood of the continuations")
        calibrated_scores = calibrate_nll(all_data["test"][0], device, all_data["domain_prompt"] if not args.empty_domain_prompt else "", tokenizer, model)
        logger.info(f"calibrate log likelihood scores: {calibrated_scores.tolist()}")

    metrics = defaultdict(list)
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stop = []
    stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
    stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
    if "llama" in args.model_name_or_path.lower():
        stop_token_ids.remove(tokenizer.unk_token_id)

    # for summarization tasks the newline char isn't used as a stop token
    if "nnl" in args.tag:
        stop_token_ids.remove(13)

    output_data = []

    with torch.inference_mode():
        model.eval()

        # TODO: replace the loop with dataloader, we need to test what the returned data looks like with the default collator, hopefully we don't need to do anything
        torch_dataset = TestItemDataset(args, all_data, dataset, tokenizer)
        dataloader = DataLoader(torch_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)
        for idx, batch in enumerate(tqdm(dataloader)):
            inputs = batch[0]
            data = inputs.pop("original_data")
            test_item = inputs.pop("test_item")

            # can't call .to() on the num_context for replug (a int)
            prefix_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs["prefix_inputs"].items()}

            if test_item.task == "generate":
                if args.model_class == "replug":
                    model.separate_forward = prefix_input["input_ids"].shape[1] > 1024

                outputs = model.generate(
                    **prefix_input,
                    max_new_tokens=args.generation_max_length,
                    min_new_tokens=args.generation_min_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=stop_token_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                prediction = tokenizer.decode(
                    outputs.sequences[0][prefix_input["input_ids"].size(1):],
                    skip_special_tokens=True,
                )
                prediction = prediction.strip()

                em = drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, test_item.answer)
                metrics["exact_match"].append(int(em))
                f1 = drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, test_item.answer)
                metrics["f1"].append(f1)
                sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, test_item.answer)
                metrics["substring_exact_match"].append(int(sub_em))

                if all_data["use_rouge"]:
                    rougel = drqa_metric_max_over_ground_truths(lambda x, y: r_scorer.score(x, y)["rougeL"].fmeasure, prediction, test_item.answer)
                    metrics["rouge-l"].append(rougel)

            elif test_item.task == "loglikelihood":
                # scores are the negative log likelihoods
                scores = []

                outputs = model(**prefix_input, use_cache=True)
                past_key_values = outputs.past_key_values
                outputs.logits = outputs.logits.to("cpu")
                # need to keep the last token logits from the prefix to predict the first token in the continuation
                last_token_logits = outputs.logits[0, -1:]
                lengths = []

                for i, continuation in enumerate(inputs["continuation_inputs"]):
                    torch.cuda.empty_cache()

                    continuation_length = continuation.pop("continuation_length")
                    continuation_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in continuation.items()}
                    if args.model_class == "cepe":
                        continuation_input["encoder_hidden_states"] = outputs.encoder_hidden_states

                    # we need to prepend the mask from the prefix
                    continuation_input["attention_mask"] = torch.concatenate([prefix_input["attention_mask"], continuation_input["attention_mask"]], dim=1)

                    continuation_outputs = model(
                        **continuation_input,
                        past_key_values=past_key_values,
                    )
                    logits = continuation_outputs.logits.detach().cpu()[0]
                    logits = torch.cat([last_token_logits, logits[:-1]], dim=0)
                    assert logits.size(0) == continuation_length

                    ids = continuation_input["input_ids"].detach().cpu()[0]
                    prob = F.softmax(logits, dim=-1)
                    # note that we can either take the mean or the sum as the score
                    score = prob[torch.arange(continuation_length), ids]
                    score = torch.log(score).sum()
                    scores.append(score)
                    continuation_input = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in continuation.items()}
                    lengths.append(continuation_length)

                scores = torch.tensor(scores)
                lengths = torch.tensor(lengths)

                prediction = scores.argmax().item()
                prediction = test_item.continuations[prediction].strip()

                # not normalized for length -- sum of loglikelihood
                prediction = []
                pred, acc = nll_acc(scores, test_item.answer_idx)
                prediction.append(pred)
                metrics["acc"].append(acc)

                # normalized for length -- average of loglikelihood
                pred, acc = nll_acc_norm(scores, test_item.answer_idx, lengths)
                prediction.append(pred)
                metrics["acc_norm"].append(acc)

                if args.calibrate_nll:
                    if all_data["recalibrate_every"]:
                        calibrated_scores = calibrate_nll(data, device, all_data["domain_prompt"] if not args.empty_domain_prompt else "", tokenizer, model)
                    pred, acc = nll_acc_calibrated(scores, test_item.answer_idx, calibrated_scores)
                    prediction.append(pred)
                    metrics["acc_calibrated"].append(acc)

                    pred, acc = nll_acc_calibrated_norm(scores, test_item.answer_idx, calibrated_scores, lengths)
                    metrics["acc_calibrated_norm"].append(acc)
                    prediction.append(pred)

            # print out some examples
            if idx < 5 or args.debug:
                logger.info(f"Example {idx+1}: ")
                logger.info(f"Input length: {len(prefix_input['input_ids'][0])}")
                if prefix_input.get("encoder_input_ids", None) is not None:
                    logger.info("-"*20)
                    logger.info(f"Encoder input shape: {prefix_input['encoder_input_ids'][0].shape}")
                    logger.info(f"Encoder inputs:\n")
                    for i, s in enumerate(tokenizer.batch_decode(prefix_input['encoder_input_ids'][0])):
                        if i >= 20:
                            logger.info("omitting the rest of the encoder inputs...")
                            break
                        logger.info(f"{s}\n")
                    logger.info("-"*50)
                logger.info(f"Decoder inputs:\n{tokenizer.decode(prefix_input['input_ids'][0])}")

                if test_item.task == "loglikelihood":
                    for i, continuation in enumerate(inputs["continuation_inputs"]):
                        logger.info(f"Continuation {i}:{tokenizer.decode(continuation['input_ids'][0])}")

                logger.info(f"Answer: {test_item.answer}")
                logger.info(f"Output: {prediction}")
            
            if args.debug:
                import pdb; pdb.set_trace()

            prefix_input = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs["prefix_inputs"].items()}

            data["output"] = prediction
            output_data.append(copy.deepcopy(data))

    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")

    averaged_metrics = {k: np.mean(v)*100 for k, v in metrics.items()}
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")

    output = {
        "args": args.__dict__,
        "data": output_data,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
    }

    if args.output_dir is not None:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        logger.info(f"done, results are written to {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--tag", type=str, default="eval", help="tag to add to the output file")

    # model setting
    parser.add_argument("--model_class", type=str, default="context", choices=["cepe", "replug", "vanilla"])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--demo_files", type=str, default=None)
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to include the title in the prompt")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # evaluation settings
    parser.add_argument("--shot", type=int, default=5, help="total number of demos (encoder + decoder)")
    parser.add_argument("--n_shot_encoder", type=int, default=0, help="number of demos to use per encoder context")
    parser.add_argument("--n_shot_decoder", type=int, default=5, help="number of demos to use in the decoder")

    parser.add_argument("--n_demo_doc", type=int, default=0, help="number of documents to use for the demo in the decoder")
    parser.add_argument("--n_demo_doc_encoder", type=int, default=0, help="number of documents to use for the test passages in the encoder input")
    parser.add_argument("--n_demo_doc_decoder", type=int, default=0, help="number of documents to use for the test passages in the encoder input")
    parser.add_argument("--encoder_demo_n_doc", type=int, default=0, help="number of documents to use for the test passages in the encoder input")

    parser.add_argument("--n_test_doc", type=int, default=0, help="number of documents to use for the test passage in the decoder")
    parser.add_argument("--n_test_doc_encoder", type=int, default=0, help="number of documents to use for the test passages in the encoder input")
    parser.add_argument("--n_test_doc_decoder", type=int, default=0, help="number of documents to use for the test passages in the encoder input")

    parser.add_argument("--context_max_length", type=int, default=256, help="max length (in number of tokens) of the context (demo and/or passages) for context models")
    parser.add_argument("--input_max_length", type=int, default=4096, help="the maximum number of tokens of the input, we truncate from the left")
    parser.add_argument("--include_title", action="store_true", help="whether to include the title in the prompt")
    parser.add_argument("--context_strategy", type=str, default="separate", help="""
        the strategy for formatting the prompt, could contain the following:
         - separate (each demo/passage are in separate encoder forward passes),
         - concat (concatenate all demos/passes together),
         - include_all (include the instruction and test question or demo QA in the encoder input),
         - include_question (append the question to the encoder input),
         - include_query (prepend the search query to the encoder input),
         - passage_at_front (put all passages at the front of the input)
    """)
    parser.add_argument("--calibrate_nll", action="store_true", help="calibrate the log likelihood of the options without context (when applicable)")
    parser.add_argument("--empty_domain_prompt", action="store_true", help="always use empty string as the domain prompt (when applicable)")
    parser.add_argument("--use_instruction", action="store_true", help="add instruction to the prompt")

    # generation settings
    parser.add_argument("--do_sample", action="store_true", help="whether to use sampling (false is greedy)")
    parser.add_argument("--generation_max_length", type=int, default=10, help="max number of tokens to generate")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")

    # model specific settings
    parser.add_argument("--replug_passage_temperature", type=float, default=1.0, help="replug passage temperature (1 is default)")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bfloat16 -- use fp32 instead")
    parser.add_argument("--debug", action="store_true", help="for debugging")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"using device {device}")

    if args.output_dir is None:
        logger.warning("no output directory specified, setting it to args.model_name_or_path but may cause error")
        args.output_dir = args.model_name_or_path
        args.separate_forward = True

    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.is_decoder = True
    if args.model_class == "replug":
        config.replug_passage_temperature = args.replug_passage_temperature

    tokenizer_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path
    logger.info(f"loading tokenizer from {tokenizer_path}")
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = config.max_position_embeddings
    if args.input_max_length < tokenizer.model_max_length:
        logger.info(f"setting tokenizer.model_max_length to {args.input_max_length}")
        tokenizer.model_max_length = args.input_max_length

    if args.model_class == "cepe":
        model_cls = LlamaForCausalContextLM
    elif args.model_class == "replug":
        model_cls = LlamaForReplugCausalLM
    elif args.model_class == "vanilla":
        model_cls = LlamaForCausalLM

    config._flash_attn_2_enabled = True
    logger.info(f"loading model from {args.model_name_or_path}")
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        config=config,
        max_memory=get_max_memory(),
        torch_dtype=torch.bfloat16 if not args.no_bf16 else torch.float32,
    )
    model = model.to(device)
    model.eval()

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    demo_files = args.demo_files.split(",")
    assert len(test_files) == len(demo_files)

    for dataset, test_file, demo_file in zip(datasets, test_files, demo_files):
        output_path = run_test(args, tokenizer, model, device, dataset, test_file, demo_file)

        if "alce" in dataset and (not os.path.exists(output_path+".score") or args.overwrite):
            import eval_alce
            logger.info("running eval_alce.py...")
            if "asqa" in dataset:
                eval_alce.main(["--f", output_path, "--citations", "--mauve"])
            elif "eli5" in dataset:
                eval_alce.main(["--f", output_path, "--citations", "--mauve", "--claims_nli"])
            elif "qampari" in dataset:
                eval_alce.main(["--f", output_path, "--citations"])

if __name__ == "__main__":
    main()
