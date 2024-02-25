import json
import logging
import math
import os
import sys
import subprocess

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed as dist

import datasets
import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    Trainer,
    LlamaTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    TrainingArguments,
    default_data_collator,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import streaming
from data import ReplugDataCollator, CombineStreamingDataset, ContextDataCollator
from modeling_llama_flash import LlamaForCausalContextLM, LlamaForReplugCausalLM, LlamaEncoder

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import signal
class SIGUSR1Callback(transformers.TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            logger.warn("Setting should save and should stop")
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    num_cross_attn_layers: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum number of cross attention layers to add, starting from the end of the model."
        },
    )
    num_cross_attn_hidden_states: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of hidden states to use from the encoder, this should be either 1 (using the last state) or equal to num_cross_attn_layers (using the corresponding hidden states)",
        },
    )
    init_mode: Optional[str] = field(
        default="copy",
        metadata={
            "help": "How to initialize the weights of the cross attention layers. Options are: 'copy' (default), 'zero', 'normal', 'none'"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    model_class: Optional[str] = field(
        default="context",
        metadata={"help": "The model class to use during instantiation. Options are: 'cepe' (default), 'vanilla', and 'replug'"}
    )
    replug_passage_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature for the retrieval scores when using replug."}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    train_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to train the encoder or not."},
    )
    train_everything: bool = field(
        default=False,
        metadata={"help": "Whether to train all parameters or not."},
    )
    encode_mode: Optional[str] = field(
        default="context_only",
        metadata={"help": "The encode mode. Options are: 'context_only' (default), 'with_query'"},
    )
    train_batch_mode: Optional[str] = field(
        default="none",
        metadata={"help": "The train batch mode. Options are: 'none' (default), 'in_batch_negative'"},
    )
    encoder_config: Optional[str] = field(
        default=None,
        metadata={"help": "Config for the encoder in case we are not using a pre-trained encoder."},
    )
    encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The encoder path, overwrite the existing encoder in the model. If set to None, then the encoder is the model itself."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    lm_loss_cof: Optional[float] = field(
        default=1.0,
        metadata={"help": "The coefficient for the LM loss."},
    )
    kl_loss_cof: Optional[float] = field(
        default=0.0,
        metadata={"help": "The coefficient for the KL loss."},
    )
    kl_loss_mode: Optional[str] = field(
        default="smooth_1e-6",
        metadata={"help": "The mode for the KL loss. Options are: 'smooth' (default), 'hard'"},
    )
    offload_hidden_states: bool = field(
        default=False,
        metadata={"help": "Whether to offload the hidden states to CPU or not."},
    )
    replug_separate_forward: bool = field(
        default=False,
        metadata={"help": "Whether to use separate forward for replug or not."},
    )

    def __post_init__(self):
        assert self.num_cross_attn_hidden_states == 1 or self.num_cross_attn_hidden_states == self.num_cross_attn_layers, "num_cross_attn_hidden_states must be either 1 or equal to num_cross_attn_layers"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    tag: Optional[str] = field(default="", metadata={"help": "Tag for the run."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data (mds dir)."})
    train_file_distill: Optional[str] = field(default=None, metadata={"help": "The input training data for distillation (mds dir)."})
    train_file_retrieval: Optional[str] = field(default=None, metadata={"help": "The input training data file for retrieval (mds dir)."})
    retrieval_mode: Optional[str] = field(default="no_neighbor", metadata={"help": "The retrieval mode. Options are: 'no_neighbor' (default), 'joint', 'separate'"})
    train_domains: Optional[str] = field(
        default="arxiv,book,c4-rp,cc,github,stackexchange,wiki",
        metadata={"help": "the domain to use for train separated by commas, RedPajama contains: {arxiv,book,c4-rp,cc,github,stackexchange,wiki}"}
    )
    train_load_strategy: Optional[str] = field(default="best", metadata={"help": "How to load the train data."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on."},
    )
    validation_file_distill: Optional[str] = field(
        default=None, metadata={"help": "The input validation data for distillation (mds dir)."}
    )
    validation_file_retrieval: Optional[str] = field(
        default=None, metadata={"help": "The input validation data file for retrieval (mds dir)."}
    )
    validation_domains: Optional[str] = field(
        default="",
        metadata={"help": "the domain to use for validation separated by commas, RedPajama contains: {arxiv,book,c4-rp,cc,github,stackexchange,wiki}"}
    )
    validation_load_strategy: Optional[str] = field(
        default="best", metadata={"help": "How to load the validation data."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    eval_window: Optional[int] = field(
        default=256,
        metadata={"help": "The number of tokens at the end of the sequence to calculate the perplexity over. Set to  0 or None to calculate the perplexity over the entire sequence."},
    )
    eval_results_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional file to write the evaluation results to."},
    )
    keep_context_mask_in_memory: bool = field(default=True, metadata={"help": "keep mask in memory or create at get item (assume the mask is all 1s)"})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    chunk_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    num_context: Optional[int] = field(default=8)
    context_size: Optional[int] = field(default=2048)
    mask_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Probability of masking a context during training."},
    )
    mask_seq_prob: Optional[float] = field(
        default=0.0, 
        metadata={"help": "Probability of masking the entire sequence if there is mask at all during training."}
    )
    maximize_data: bool = field(
        default=False,
        metadata={"help": "Maximize the amount of data from the preprocessing data, only applies to training."},
    )
    save_to_s3: bool = field(
        default=False,
        metadata={"help": "Save the model to s3."},
    )
    s3_root_path: Optional[str] = field(
        default=None,
        metadata={"help": "The root path to save the model to s3."},
    )
    overwrite_eval_file: bool = field(
        default=False,
        metadata={"help": "Overwrite the evaluation file."},
    )

    #def __post_init__(self):
    #    if self.dataset_name is None and self.train_file is None and self.validation_file is None:
    #        raise ValueError("Need either a dataset name or a training/validation file.")
    #    else:
    #        if self.train_file is not None:
    #            extension = self.train_file.split(".")[-1]
    #            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
    #        if self.validation_file is not None:
    #            extension = self.validation_file.split(".")[-1]
    #            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

## This function sync a folder to s3
def save_to_s3(local_path, s3_path):

    if not dist.is_initialized() or dist.get_rank() == 0:

        cmd = [
            "aws", "s3", "sync", local_path, s3_path
        ]

        print(f"Uploading {local_path} to {s3_path}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Folder {local_path} uploaded to {s3_path}")
        except subprocess.CalledProcessError as e:
            # Handle errors in the called subprocess
            print(f"An error occurred: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An unexpected error occurred: {e}")

def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    self._original_save_checkpoint(model, trial, metrics=metrics)

    if getattr(self.args, "save_to_s3", False):
        s3_path = self.args.s3_root_path + os.path.join(os.path.basename(run_dir), checkpoint_folder)
        save_to_s3(output_dir, s3_path)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_file_flag="--config")

    if data_args.save_to_s3:
        training_args.save_to_s3 = True
        training_args.s3_root_path = data_args.s3_root_path

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    streaming.base.util.clean_stale_shared_memory()

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome. We will overwrite the output_dir by default."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # load tokenizer and config
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    config.is_decoder = True
    config._flash_attn_2_enabled = True
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if training_args.do_train:
        # load the training dataset
        domains = data_args.train_domains
        logger.info(f"loading train dataset with domains {domains}")

        train_dataset = CombineStreamingDataset(
            data_args.train_file,
            distill_remote=data_args.train_file_distill,
            retrieval_remote=data_args.train_file_retrieval,
            retrieval_mode=data_args.retrieval_mode,
            num_context=data_args.num_context,
            context_size=data_args.context_size,
            chunk_size=data_args.chunk_size,
            domains=domains,
            load_strategy=data_args.train_load_strategy,
            tokenizer=tokenizer,
            epoch_size=data_args.max_train_samples,
            mask_prob=data_args.mask_prob,
            mask_seq_prob=data_args.mask_seq_prob,
        )

        if data_args.train_file_distill is not None:
            config.kl_loss_cof = model_args.kl_loss_cof
            config.kl_loss_mode = model_args.kl_loss_mode

        logger.info(f"loaded train dataset size: {len(train_dataset)}")

        # logger.info(f"printing out some examples of the train dataset")
        # for i in range(min(3, len(train_dataset))):
        #     d = train_dataset[i]
        #     logger.info(f"input ids: {d['input_ids']}; input text: {tokenizer.decode(d['input_ids'])}")
        #     if "encoder_input_ids" in d:
        #         ids = d["encoder_input_ids"]
        #         logger.info(f"context input ids: {ids}; context text: {tokenizer.batch_decode(ids)}")

    if training_args.do_eval:
        # load the eval dataset
        domains = data_args.validation_domains
        logger.info(f"loading validation dataset with domains {domains}")

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
            tokenizer=tokenizer,
            epoch_size=data_args.max_eval_samples,
        )

        logger.info(f"loaded eval dataset size: {len(eval_dataset)}")


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1, labels.shape[-1])
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        preds = preds[labels != -100]
        labels = labels[labels != -100]
        results = metric.compute(predictions=preds, references=labels)
        return results

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

    else:
        raise NotImplementedError(f"Model class {model_args.model_class} not implemented")

    encoder = None
    # load the encoder if we have one
    if model_args.encoder_name_or_path is not None:
        logger.info(f"Loading encoder from {model_args.encoder_name_or_path}")
        logger.info("Note that we assume the encoder has the same tokenizer as the model")
        encoder_config = LlamaConfig.from_pretrained(model_args.encoder_name_or_path)
        encoder_config._flash_attn_2_enabled = config._flash_attn_2_enabled
        config.encoder_hidden_size = encoder_config.hidden_size
        config.encoder_config = encoder_config.to_dict()

        encoder = LlamaEncoder.from_pretrained(
            model_args.encoder_name_or_path,
            config=encoder_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Loaded encoder config: {encoder_config}")
        logger.info(f"Loaded encoder: {encoder}")
        logger.info(f"Total number of parameters in encoder model: {sum(p.numel() for p in encoder.parameters())}")

    elif model_args.encoder_config is not None:
        logger.info(f"Loading encoder from config for random initialization {model_args.encoder_config}")
        encoder_config = LlamaConfig.from_pretrained(model_args.encoder_config)
        encoder_config._flash_attn_2_enabled = config._flash_attn_2_enabled
        config.encoder_hidden_size = encoder_config.hidden_size
        config.encoder_config = encoder_config.to_dict()
        encoder = LlamaEncoder._from_config(encoder_config, torch_dtype=torch_dtype)
        logger.info(f"Loaded encoder config: {encoder_config}")
        logger.info(f"Loaded encoder: {encoder}")
        logger.info(f"Total number of parameters in encoder model: {sum(p.numel() for p in encoder.parameters())}")

    # instantiate model, initialize weights and set the encoder we want to use (not used rn)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        use_auth_token=True,
    )
    if encoder is not None:
        model.set_encoder(encoder)

    def initialize_cross_attention_weights(model):
        if model_args.init_mode.lower() == "none":
            return
        for l in model.model.layers:
            if l.do_cross_attention:
                l.cross_attn.q_proj.weight.data = l.self_attn.q_proj.weight.data.clone()
                l.cross_attn.k_proj.weight.data = l.self_attn.k_proj.weight.data[:l.cross_attn.k_proj.out_features, :l.cross_attn.k_proj.in_features].clone()
                l.cross_attn.v_proj.weight.data = l.self_attn.v_proj.weight.data[:l.cross_attn.v_proj.out_features, :l.cross_attn.v_proj.in_features].clone()
                if model_args.init_mode == "copy":
                    l.cross_attn.o_proj.weight.data = l.self_attn.o_proj.weight.data.clone()
                elif model_args.init_mode == "zero":
                    torch.nn.init.zeros_(l.cross_attn.o_proj.weight.data)
                elif model_args.init_mode == "normal":
                    torch.nn.init.kaiming_normal_(l.cross_attn.o_proj.weight.data)
                l.cross_attn.layernorm.weight.data = l.post_attention_layernorm.weight.data.clone()

    if training_args.do_train and model_args.model_class == "cepe":
        logger.info(f"Initializing cross attention weights with mode {model_args.init_mode}")
        initialize_cross_attention_weights(model)

    logger.info(f"Config: {config}")
    logger.info(f"Model: {model}")
    logger.info(f"Total number of parameters in model: {sum(p.numel() for p in model.parameters())}")

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    def train_param(param_name):
        if model_args.train_everything:
            return True
        if model_args.train_encoder and "encoder" in param_name:
            return True
        return "cross_attn" in param_name if model_args.model_class == "cepe" and config.num_cross_attn_layers > 0 else True

    for n, p in model.named_parameters():
        p.requires_grad = train_param(n)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.add_callback(SIGUSR1Callback)

    trainer._original_save_checkpoint = trainer._save_checkpoint
    trainer._save_checkpoint = _save_checkpoint.__get__(trainer, Trainer)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info("Starting train")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        logger.info("Finished training")
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_results_file = f"{training_args.output_dir}/eval-{data_args.tag}-chunk_size{data_args.chunk_size}-n_ctx{data_args.num_context}-ctx_size{data_args.context_size}-domain{data_args.validation_domains}-sample{data_args.max_eval_samples}-eval_window{data_args.eval_window}-load_strategy{data_args.validation_load_strategy}-ret_mode{data_args.retrieval_mode}.json" if data_args.eval_results_file is None else data_args.eval_results_file

        if os.path.exists(eval_results_file) and not data_args.overwrite_eval_file:
            logger.info(f"Evaluation results file already exists at {eval_results_file}, skipping evaluation")
            exit()

        logger.info("Starting evaluation")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        metrics["num_params"] = sum(p.numel() for p in model.parameters())
        metrics["eval_window"] = data_args.eval_window
        metrics["num_context"] = data_args.num_context
        metrics["context_size"] = data_args.context_size
        metrics["chunk_size"] = data_args.chunk_size
        metrics["validation_file"] = data_args.validation_file
        metrics["num_eval_samples"] = len(eval_dataset)
        metrics["validation_domains"] = data_args.validation_domains
        print(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info(f"Saving evaluation results to {eval_results_file}")
        with open(eval_results_file, "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
