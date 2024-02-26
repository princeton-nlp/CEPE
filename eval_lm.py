import os 
import sys

import json
import logging
import math
import time
from dataclasses import field, dataclass
from typing import Optional

from tqdm import tqdm, trange
import numpy as np 
import torch

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer, 
    TrainingArguments, 
    HfArgumentParser,
)
from transformers.testing_utils import CaptureLogger
import datasets

from train import ModelArguments, DataTrainingArguments
from data import CombineStreamingDataset, ContextDataCollator, ReplugDataCollator
from dataset_utils import load_lm_dataset, add_contriever_scores

from modeling_llama_flash import LlamaForCausalContextLM, LlamaForReplugCausalLM

logger = logging.getLogger(__name__)

@dataclass
class ModelArgumentsEval(ModelArguments):
    cache_start_size: Optional[int] = field(
        default=4, metadata={"help": "Start size for the KV cache in StreamingLLM"},
    )
    cache_recent_size: Optional[int] = field(
        default=2044, metadata={"help": "Recent size for the KV cache in StreamingLLM"},
    )
    enable_positional_shift: Optional[bool] = field(
        default=False, metadata={"help": "Enable positional shift for StreamingLLM"},
    )
    eval_step_size: Optional[int] = field(
        default=2048, metadata={"help": "Step step for evaluation in StreamingLLM (number of tokens evaluated at a time)"},
    )
    shard_id: Optional[int] = field(
        default=0, metadata={"help": "Shard id for evaluation in StreamingLLM"},
    )
    num_shards: Optional[int] = field(
        default=1, metadata={"help": "Number of shards for evaluation in StreamingLLM"},
    )
    filter_length: Optional[int] = field(
        default=32768, metadata={"help": ""},
    )


def main():
    parser = HfArgumentParser((ModelArgumentsEval, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_file_flag="--config")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    if "Yarn" in model_args.model_name_or_path:
        sys.path.append(model_args.model_name_or_path)
        from configuration_llama import LlamaConfig
        from modeling_llama_together_yarn import LlamaForCausalLM
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    config._flash_attn_2_enabled = True
    config.is_decoder = True
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path if model_args.tokenizer_name is None else model_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)

    # find the appropriate model cls
    if model_args.model_class == "cepe":
        logger.info("Using modified Llama")
        model_cls = LlamaForCausalContextLM
        collator = ContextDataCollator()
        config.lm_loss_cof = model_args.lm_loss_cof

        if not hasattr(config, "num_cross_attn_layers"):
            logger.info(f"Config does not have cross attention set (assuming we are starting with original Llama checkpoint), using model_args: {model_args.num_cross_attn_layers}")
            config.num_cross_attn_layers = model_args.num_cross_attn_layers
            config.num_cross_attn_hidden_states = model_args.num_cross_attn_hidden_states
            config.do_cross_attention = True
            config.encoder_is_model = model_args.encoder_name_or_path is None and model_args.encoder_config is None
            config.train_encoder = model_args.train_encoder

        # we always overwrite these two configs
        config.encode_mode = model_args.encode_mode
        config.train_batch_mode = model_args.train_batch_mode
        config.offload_hidden_states = model_args.offload_hidden_states

    elif model_args.model_class == "vanilla":
        logger.info("Using vanilla Llama")
        model_cls = LlamaForCausalLM
        collator = ContextDataCollator()

    elif model_args.model_class == "replug":
        logger.info("Using replug Llama")
        model_cls = LlamaForReplugCausalLM
        collator = ReplugDataCollator()
        config.replug_passage_temperature = model_args.replug_passage_temperature
        config.separate_forward = model_args.replug_separate_forward
        config.lm_eval_mode = True

    elif model_args.model_class == "streamingllm":
        logger.info("Using attention sink")
        sys.path.append(os.path.join(os.getcwd(), "streaming-llm"))
        from streaming_llm.kv_cache import StartRecentKVCache
        model_cls = LlamaForCausalLM
        collator = ContextDataCollator()
        config._flash_attn_2_enabled = True

        k_seq_dim = v_seq_dim = 2
        kv_cache = StartRecentKVCache(
            start_size=model_args.cache_start_size,
            recent_size=model_args.cache_recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )

    if model_args.filter_length > 32768:
        device_map = "balanced"
    else:
        device_map = "auto"

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config, 
        torch_dtype=torch_dtype,
        use_auth_token=True,
        device_map=device_map,
    )
    # model = model.to(device) ## shouldn't need this with device map 
    logger.info(f"Loaded model: {model}")

    if model_args.model_class == "streamingllm" and model_args.enable_positional_shift:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention
        enable_llama_pos_shift_attention(model)

    if training_args.do_eval:
        domains = data_args.validation_domains
        logger.info(f"loading validation dataset with domains {domains}")

        if os.path.exists(data_args.validation_file):
            # note that we can also load from a remote file (s3), but in this case we assume it's a local file
            eval_dataset = CombineStreamingDataset(
                data_args.validation_file,
                distill_remote=data_args.validation_file_distill,
                retrieval_remote=data_args.validation_file_retrieval,
                retrieval_mode=data_args.retrieval_mode,
                num_context=data_args.num_context,
                context_size=data_args.context_size,
                chunk_size=data_args.chunk_size,
                loss_chunk_size=data_args.eval_window,
                domains=domains,
                load_strategy=data_args.validation_load_strategy,
                prepend_index=data_args.prepend_index,
                tokenizer=tokenizer,
                epoch_size=data_args.max_eval_samples,
            )

        else:
            # otherwise we load from huggingface
            dataset, text_column_name = load_lm_dataset(data_args.validation_file)
            tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                # clm input could be much much longer than block_size
                if "Token indices sequence length is longer than the" in cl.out:
                    tok_logger.warning(
                        "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                        " before being passed to the model."
                    )
                return output
            
            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=dataset.column_names,
                    desc="Running tokenizer on dataset",
                )
                tokenized_datasets = tokenized_datasets.filter(lambda example: len(example["input_ids"]) >= model_args.filter_length)

            def preprocess(examples):
                # first calculate out the block size from n_ctx, ctx_size, and chunk_size and filter by length
                # second take chunks out with a stride of the eval sliding window
                # finally put things into the input_ids and encoder_input_ids if needed
                results = []
                for idx in range(len(examples["input_ids"])):
                    input_ids = examples["input_ids"][idx]
                    attention_mask = examples["attention_mask"][idx]

                    if len(input_ids) < model_args.filter_length:
                        continue
                    stride = data_args.eval_window
                    total_length = data_args.num_context * data_args.context_size + data_args.chunk_size

                    for i in range(0, len(input_ids) - total_length, stride):
                        # we don't need the mask because we tokenized sequences without padding (the collator/forward func will handle the mask)
                        ids = np.array(input_ids[i:i+total_length], dtype=np.int32)

                        if "put_in_decoder" in data_args.validation_load_strategy:
                            results.append({"input_ids": ids,})
                        else:
                            encoder_input_ids = ids[:data_args.num_context * data_args.context_size].reshape(data_args.num_context, data_args.context_size)
                            ids = ids[data_args.num_context * data_args.context_size:]
                            results.append({
                                "input_ids": ids, 
                                "encoder_input_ids": encoder_input_ids, 
                            })
                        labels = np.copy(ids).astype(np.int32)
                        if stride < total_length:
                            labels[:-stride] = -100
                        results[-1]["labels"] = labels

                        if model_args.filter_length > 32768:
                            # only keep one sequence per document if the length is too long
                            # otherwise we might store a lot of tokens with sliding window --> oom
                            break
                        
                results = {k: np.stack([d[k] for d in results]) for k in results[0]}
                return results
            
            with training_args.main_process_first(desc="dataset preprocess"):
                # since the test sets don't contain many sequences after the length filter,
                # using the default batch_size (1000) can be very slow and may even give oom 
                # since we are also using sliding window (feel free to change for your system)
                eval_dataset = tokenized_datasets.map(
                    preprocess,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=tokenized_datasets.column_names,
                    batch_size=32,
                    desc="Running preprocessing on dataset",
                )
            logger.info(f"eval dataset size after filtering: {len(eval_dataset)}")
                
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            eval_dataset = add_contriever_scores(eval_dataset, tokenizer) if model_args.model_class == "replug" else eval_dataset

        logger.info(f"loaded eval dataset size: {len(eval_dataset)}")

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        eval_results_file = f"{training_args.output_dir}/eval-{data_args.tag}-chunk_size{data_args.chunk_size}-n_ctx{data_args.num_context}-ctx_size{data_args.context_size}-domain{data_args.validation_domains}-sample{data_args.max_eval_samples}-eval_window{data_args.eval_window}-load_strategy{data_args.validation_load_strategy}-ret_mode{data_args.retrieval_mode}{('-shard'+str(model_args.shard_id)) if model_args.num_shards > 1 else ''}.json" if data_args.eval_results_file is None else data_args.eval_results_file

        if os.path.exists(eval_results_file) and not data_args.overwrite_eval_file:
            logger.info(f"Evaluation results file already exists at {eval_results_file}, exiting evaluation...")
            exit()

        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info("***** Running evaluation *****")
        model.eval()
        step_size = model_args.eval_step_size

        loss_fct = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            all_losses = []
            shard_start = 0
            shard_end = len(eval_dataset)
            if model_args.num_shards > 1:
                shard_size = len(eval_dataset) // model_args.num_shards
                shard_start = model_args.shard_id * shard_size
                shard_end = (model_args.shard_id + 1) * shard_size if model_args.shard_id < model_args.num_shards - 1 else len(eval_dataset)

            start_time = time.time()
            pbar = tqdm(eval_dataloader)
            for idx, batch in enumerate(pbar):
                if idx >= data_args.max_eval_samples or idx >= shard_end:
                    break
                if idx < shard_start:
                    continue

                batch = {k: v.to(device) for k, v in batch.items()}

                if model_args.model_class == "streamingllm":
                    # from https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py
                    past_key_values = None
                    seq_length = batch["input_ids"].shape[1]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    total_loss = torch.zeros(1).to(device)

                    # optimization: we should run forward with the initial window for the start of the sequence
                    # this shouldn't change the performance but avoid unnecessary iterations
                    initial_window = model_args.cache_start_size + model_args.cache_recent_size
                    input_ids = batch["input_ids"][:, :initial_window]
                    attention_mask = batch["attention_mask"][:, :initial_window] if "attention_mask" in batch else None
                    labels = batch["labels"][:, : initial_window]
                    outputs = model(
                        input_ids, 
                        labels=labels,
                        past_key_values=past_key_values, 
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    past_key_values = kv_cache(past_key_values)

                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                    shift_labels = shift_labels.view(-1).to(shift_logits.device)
                    last_logit = outputs.logits[:, -1:, :]

                    loss = loss_fct(shift_logits, shift_labels)
                    total_loss += loss[shift_labels != -100].sum()

                    for i in range(initial_window, seq_length, step_size):
                        input_ids = batch["input_ids"][:, i : i + step_size]
                        labels = batch["labels"][:, i : i + step_size]
                        outputs = model(input_ids, past_key_values=past_key_values, labels=labels, use_cache=True)
                        past_key_values = outputs.past_key_values
                        past_key_values = kv_cache(past_key_values)

                        shift_logits = torch.cat((last_logit, outputs.logits[..., :-1, :]), dim=-2).contiguous()
                        shift_labels = labels[..., :].contiguous()
                        shift_logits = shift_logits.view(-1, model.config.vocab_size)
                        shift_labels = shift_labels.view(-1).to(shift_logits.device)
                        last_logit = outputs.logits[:, -1:, :]

                        loss = loss_fct(shift_logits, shift_labels)
                        total_loss += loss[shift_labels != -100].sum()
                    
                    loss = total_loss / data_args.eval_window

                else:
                    outputs = model(**batch)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()

                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                    shift_labels = shift_labels.view(-1).to(shift_logits.device)

                    loss = loss_fct(shift_logits, shift_labels)

                pbar.set_description(f"Loss: {loss.item()}")
                all_losses.append(loss.item())

            eval_time = time.time() - start_time 
            mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
            eval_loss = torch.tensor(all_losses).mean().item()
            logger.info(f"Eval loss: {eval_loss}")

        metrics = {"eval_loss": eval_loss}
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"Perplexity: {perplexity}")
        metrics["perplexity"] = perplexity
        metrics["num_params"] = sum(p.numel() for p in model.parameters())
        metrics["eval_window"] = data_args.eval_window
        metrics["num_context"] = data_args.num_context
        metrics["context_size"] = data_args.context_size
        metrics["chunk_size"] = data_args.chunk_size
        metrics["validation_file"] = data_args.validation_file
        metrics["num_eval_samples"] = len(eval_dataset)
        metrics["validation_domains"] = data_args.validation_domains
        metrics["memory_usage"] = mem_usage
        metrics["eval_time"] = eval_time
        metrics["eval_samples_per_second"] = len(all_losses) / eval_time
        if model_args.num_shards > 1:
            metrics["all_losses"] = all_losses
        logger.info(f"Metrics: {metrics}")

        logger.info(f"Saving evaluation results to {eval_results_file}")
        with open(eval_results_file, "w") as f:
            json.dump(metrics, f, indent=4)

    logger.info("finished...")

if __name__ == "__main__":
    main()