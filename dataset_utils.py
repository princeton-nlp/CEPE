import json
import re

from datasets import load_dataset
import transformers
from transformers.testing_utils import CaptureLogger
from utils import drqa_metric_max_over_ground_truths, drqa_exact_match_score


def load_data(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
    else:
        raise NotImplementedError(f"file format {path} not supported")
    return data


def preprocess_alce(args, demos_data, test_data):
    # given the demos data (which is the prompt file from ALCE) and the test data (which is the test file from ALCE)
    # return the data in the format that we need and also the templates

    demos = demos_data.pop("demos")
    for item in demos:
        item["answer"] = " " + item["answer"]
        item["docs"] = item.pop("docs")[:args.n_demo_doc]

    for item in test_data:
        item["docs"] = item.pop("docs")[:args.n_test_doc]

    template = demos_data

    template["template"] = template.pop("demo_prompt")
    template["document_template"] = template.pop("doc_prompt")
    template["balanced_sampling"] = False
    template["recalibrate_every"] = False
    template["truncate_seperator"] = "... [The rest of the documents is omitted]\n\n"
    # template["instruction"] = template["instruction"].split("\n\n")[0] + f" Given {args.n_test_doc} documents, the citations that you can use are " + "".join([f"[{i+1}]" for i in range(args.n_test_doc)]) + ".\n\n"
    template["use_rouge"] = False

    return demos, test_data, template


def load_qa_templates(dataset, include_title=True):
    truncate_seperator = "... [The rest of the documents is omitted]\n\n"
    if dataset == "mmlu":
        document_template = "Knowledge (Title: {title}): {text}" if include_title else "Knowledge: {text}"
        template = "{instruction}Question: {question}\nChoices:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:{answer}"
        domain_prompt = "Answer:"
        recalibrate_every = True
        balanced_sampling = False
        instruction = "Instruction: Use the knowledge to choose the correct choice to the question.\n\n"

    elif dataset == "nq" or dataset == "popqa" or dataset == "boolq" or dataset == "triviaqa":
        document_template = "Document (Title: {title}): {text}" if include_title else "Document: {text}"
        template = "{instruction}Question: {question}\nAnswer:{answer}"
        domain_prompt = "Answer:"
        recalibrate_every = False
        balanced_sampling = True if dataset == "boolq" else False
        instruction = "Instruction: Use the document(s) to write an accurate and concise answer to the question.\n\n"

    return {
        "document_template": document_template,
        "template": template,
        "domain_prompt": domain_prompt,
        "recalibrate_every": recalibrate_every,
        "balanced_sampling": balanced_sampling,
        "instruction": instruction,
        "truncate_seperator": truncate_seperator,
        "use_rouge": False, 
    }


def add_mmlu_options(data):
    for item in data:
        options = ["A", "B", "C", "D"]
        item["options"] = [f" {o}. " + item[o].strip() for o in options]
        item["answer"] = f" {item['answer']}. {item[item['answer'].strip()]}"
    return data


def add_boolq_options(data):
    for item in data:
        item["options"] = [" True", " False"]
        item["answer"] = " "+item["answer"][0]
    return data


def filter_contexts(data):
    # filter the contexts and only keep the ones that contain the answer
    new_data = []
    for d in data:
        d["ctxs"] = [ctx for ctx in d["ctxs"] if drqa_metric_max_over_ground_truths(drqa_exact_match_score, ctx["text"], d["answer"])]
        if len(d["ctxs"]) > 0:
            new_data.append(d)
    return new_data


def load_hf_dataset(dataset, train, test):
    recalibrate_every = False
    balanced_sampling = False
    truncate_seperator = "\n\n"
    document_template = None
    use_rouge = False
    domain_prompt = None

    if dataset == "ag_news":
        all_dataset = load_dataset("ag_news")
        options = [" World", " Sports", " Business", " Sci/Tech"]
        template = "{instruction}Article: {text}\nTopic:{answer}"
        instruction = "Instruction: Choose the correct topic for the article.\n\n"

        all_dataset = all_dataset.map(lambda example: {**example, "options": options, "answer": options[example["label"]]})
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["test"]

        recalibrate_every = False
        balanced_sampling = True
        domain_prompt = "Topic:"

    elif dataset == "glue/sst2" or dataset == "sst2":
        all_dataset = load_dataset("glue", "sst2")
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]
        options = [" negative", " positive"]
        template = "{instruction}Sentence: {sentence}\nSentiment:{answer}"
        instruction = "Instruction: Choose the correct sentiment for the sentence.\n\n"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options, "answer": options[example["label"]]})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options, "answer": options[example["label"]]})

        recalibrate_every = False
        balanced_sampling = True
        domain_prompt = "Sentiment:"

    elif dataset == "mr":
        train_dataset = load_dataset("rotten_tomatoes")["train"]
        test_dataset = load_dataset("rotten_tomatoes")["test"]
        options = [" negative", " positive"]
        template = "{instruction}Review: {text}\nSentiment:{answer}"
        instruction = "Instruction: Choose the correct sentiment for the review, either positive or negative.\n\n"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options, "answer": options[example["label"]]})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options, "answer": options[example["label"]]})

        domain_prompt = "Sentiment:"
        recalibrate_every = False
        balanced_sampling = True

    elif dataset == "govreport" or dataset == "scrolls/govreport":
        # the two should be the exact same
        if dataset == "govreport":
            all_dataset = load_dataset("ccdv/govreport-summarization")
            all_dataset = all_dataset.map(lambda example: {
                "ctxs": [{"text": example["report"]}], "answer": example["summary"]
            })
        else:
            all_dataset = load_dataset("tau/scrolls", "gov_report")
            all_dataset = all_dataset.map(lambda example: {"ctxs": [{"text": example["input"]}], "answer": example["output"]})

        document_template = "Report:\n{text}"
        template = "{instruction}Summary:\n{answer}"

        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]

        instruction = 'Instruction: You are given a report by a government agency. Write a one-page summary of the report.\n'
        truncate_seperator = "... [The rest of the report is omitted]\n\n"
        use_rouge = True

    elif dataset == "summ_screen_fd":
        # we use scrolls because it's nicely preprocessed and has 300+ examples in the validation set
        # (as opposed to the 20 validation examples in the ZeroScrolls set)
        all_dataset = load_dataset("tau/scrolls", "summ_screen_fd")
        document_template = "Episode Script:\n{text}"
        template = "{instruction}Summary:\n{answer}"
        
        all_dataset = all_dataset.map(lambda example: {
            **example, "ctxs": [{"text": example["input"]}], "answer": example["output"]
        })
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]

        instruction = "You are given a script of a TV episode. Summarize the episode in a paragraph.\n\n"
        truncate_seperator = "... [The rest of the episode script is omitted]\n\n"
        use_rouge = True

    elif dataset == "qmsum":
        all_dataset = load_dataset("tau/scrolls", "qmsum")
        document_template = "Transcript:\n{text}"
        template = "{instruction}Query:\n{question}\n\nAnswer:\n{answer}"

        # we need to parse out the query from the input of scrolls
        all_dataset = all_dataset.map(lambda example: {
            **example, "question": example["input"].split("\n")[0].strip(), "ctxs": [{"text": example["input"][example["input"].index("\n\n")+2:].strip()}], "answer": example["output"],
        })
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]

        instruction = "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n"
        truncate_seperator = "... [The rest of the transcript is omitted]\n\n"
        use_rouge = True

    elif dataset == "narrativeqa":
        all_dataset = load_dataset("narrativeqa")
        template = "{instruction}Question:\n{question}\n\nAnswer:\n{answer}"
        document_template = "Story:\n{text}"

        all_dataset = all_dataset.map(lambda example: {
            "question": example["question"]["text"], 
            "answer": [ex["text"] for ex in example["answers"]],
            "ctxs": [{"text": example["document"]["text"]}],
        })
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]
        instruction = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n"
        truncate_seperator = "... [The rest of the story is omitted]\n\n"

        # note: zeroscrolls uses F1 for narrativeqa
        use_rouge = True

    elif dataset == "qasper":
        # instead of using allenai/qasper, we use tau/scrolls, because it's nicely preprocessed
        # but the instructions are from zeroscrolls
        all_dataset = load_dataset("tau/scrolls", "qasper")
        template = "{instruction}Question:\n{question}\n\nAnswer:\n{answer}"
        document_template = "Article:\n{text}"

        all_dataset = all_dataset.map(lambda example: {
            "ctxs": [{"text": example["input"][example["input"].index("\n\n")+2:].strip()}], 
            "question": example["input"][:example["input"].index("\n\n")].strip(), 
            "answer": example["output"],
        })
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]

        instruction = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n\n'
        truncate_seperator = "... [The rest of the article is omitted]\n\n"

        # note: zeroscrolls use F1 for qasper
        use_rouge = True 

    elif dataset == "quality":
        all_dataset = load_dataset("tau/scrolls", "quality")
        document_template = "Story:\n{text}"
        # template from zeroscrolls, which is slighlty different from the scrolls template
        template = "{instruction}Question and Possible Answers:\n{question}\n\n{A}\n{B}\n{C}\n{D}\n\nAnswer:{answer}"

        def preprocess(example):
            input_text = example["input"]
            example["question"] = input_text.split('\n')[0]
            labels = ["A", "B", "C", "D"]
            for i, option in enumerate(labels):
                idx = input_text.index(f'({option})')
                o = " " + input_text[idx:].strip()
                if i < 3:
                    next_index = o.index(f'({labels[i+1]})')
                else:
                    next_index = o.index("\n\n")
                example[option] = o[:next_index].strip()
                input_text = input_text[idx + len(example[option]):]
            example["ctxs"] = [{'text': input_text.strip()}]
            answer = ""
            for option in labels:
                if example['output'] in example[option]:
                    answer = " " + option
                    break
            assert answer != ""
            if "generate" in dataset:
                # either just outputing the letter or the entire answer is considered correct
                example["answer"] = [answer, f'({answer.strip()}) {example["output"]}', ]
            
            else:
                example["answer"] = answer
                example["options"] = [' ' + l for l in labels]
            return example

        train_dataset = all_dataset["train"].map(preprocess)
        test_dataset = all_dataset["validation"].map(preprocess)

        instruction = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\n"
        truncate_seperator = "... [The rest of the story is omitted]\n\n"
        # zeroscrolls use acc as the metric but it's a bit harsh so we also include rouge-l
        use_rouge = True

    elif dataset == "sst5":
        all_dataset = load_dataset("SetFit/sst5")
        label_mapping = {0: ' terrible', 1: ' bad', 2: ' okay', 3: ' good', 4: ' great'}
        template = "{instruction}Sentence: {text}\nSentiment:{answer}"
        instruction = "Instruction: Choose the correct sentiment for the sentence.\n\n"

        all_dataset = all_dataset.map(lambda example: {**example, "answer": label_mapping[example["label"]], "options": label_mapping.values()})
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]
        recalibrate_every = False
        balanced_sampling = True
        domain_prompt = "Sentiment:"

    elif dataset == "trec-coarse" or dataset == "trec-fine":
        all_dataset = load_dataset("trec")
        template = "{instruction}Question: {text}\nType:{answer}"

        # from https://github.com/AI21Labs/Parallel-Context-Windows/blob/main/datasets_loader.py
        # labels mapping based on: https://aclanthology.org/2023.acl-long.352/, https://aclanthology.org/C16-1116.pdf, https://aclanthology.org/C02-1150.pdf 
        if dataset == "trec-coarse":
            label_mapping = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: 'numeric'}
            label_mapping = {k: " " + v for k, v in label_mapping.items()}
            all_dataset = all_dataset.map(lambda example: {**example, "answer": label_mapping[example["coarse_label"]], "options": label_mapping.values()})
        else:
            label_mapping = {0: 'abbreviation abbreviation', 1: 'abbreviation expansion', 2: 'entity animal', 3: 'entity body', 4: 'entity color', 5: 'entity creation', 6: 'entity currency', 7: 'entity disease', 8: 'entity event', 9: 'entity food', 10: 'entity instrument', 11: 'entity language', 12: 'entity letter', 13: 'entity other', 14: 'entity plant', 15: 'entity product', 16: 'entity religion', 17: 'entity sport', 18: 'entity substance', 19: 'entity symbol', 20: 'entity technique', 21: 'entity term', 22: 'entity vehicle', 23: 'entity word', 24: 'description definition', 25: 'description description', 26: 'description manner', 27: 'description reason', 28: 'human group', 29: 'human individual', 30: 'human title', 31: 'human description', 32: 'location city', 33: 'location country', 34: 'location mountain', 35: 'location other', 36: 'location state', 37: 'numeric code', 38: 'numeric count', 39: 'numeric date', 40: 'numeric distance', 41: 'numeric money', 42: 'numeric order', 43: 'numeric other', 44: 'numeric period', 45: 'numeric percent', 46: 'numeric speed', 47: 'numeric temperature', 48: 'numeric size', 49: 'numeric weight'}
            label_mapping = {k: " " + v for k, v in label_mapping.items()}
            all_dataset = all_dataset.map(lambda example: {**example, "answer": label_mapping[example["fine_label"]], "options": label_mapping.values()})
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["test"]
        domain_prompt = "Type:"
        recalibrate_every = False
        balanced_sampling = True
        instruction = "Instruction: Choose the correct type for the question.\n"

    elif dataset == "dbpedia":
        all_dataset = load_dataset("dbpedia_14")
        # https://github.com/AI21Labs/Parallel-Context-Windows/blob/e6d31005f22273ccd208ca10f658a14c445ebb7e/datasets_loader.py#L148
        # we ignore the title?
        options = [" Company", " School", " Artist", " Athlete", " Politics", " Transportation", " Building", " Nature", " Village", " Animal", " Plant", " Album", " Film", " Book"]
        template = "{instruction}Input: {content}\nType:{answer}"
        all_dataset = all_dataset.map(lambda example: {**example, "answer": options[example["label"]], "options": options})

        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["test"]
        domain_prompt = "Type:"
        recalibrate_every = False
        balanced_sampling = True
        instruction = "Instruction: Choose the correct type for the input.\n"

    elif dataset == "nlu_scenario" or dataset == "nlu_intent":
        all_dataset = load_dataset("nlu_evaluation_data")
        all_dataset = all_dataset["train"].train_test_split(seed=42)

        if dataset == "nlu_intent":
            labels = all_dataset["train"].features["label"].names
            labels = [" " + l.replace("_", " ") for l in labels]

            all_dataset = all_dataset.map(lambda example: {**example, "answer": labels[example["label"]], "options": labels})
            template = "{instruction}Utterance: {text}\nIntent:{answer}"
            domain_prompt = "Intent:"
            instruction = "Instruction: Choose the correct intent for the utterance.\n"

        else:
            options = [' general', ' weather', ' play', ' music', ' qa', ' audio', ' alarm', ' email', ' calendar', ' cooking', ' datetime', ' news', ' social', ' recommendation', ' iot', ' lists', ' takeaway', ' transport']
            all_dataset = all_dataset.map(lambda example: {**example, "answer": " "+example["scenario"], "options": options})
            template = "{instruction}Utterance: {text}\nScenario:{answer}"
            domain_prompt = "Scenario:"
            instruction = "Instruction: Choose the correct scenario for the utterance.\n"

        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["test"]

        recalibrate_every = False
        balanced_sampling = True

    elif dataset == "banking77":
        all_dataset = load_dataset("banking77")
        labels = all_dataset["train"].features["label"].names
        labels = [" " + l.replace("_", " ") for l in labels]

        all_dataset = all_dataset.map(lambda example: {**example, "answer": labels[example["label"]], "options": labels})
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["test"]

        template = "{instruction}Query: {text}\nIntent:{answer}"
        domain_prompt = "Intent:"
        recalibrate_every = False
        balanced_sampling = True
        instruction = "Instruction: Choose the correct intent for the query.\n"

    elif dataset == "clinic150":
        all_dataset = load_dataset("clinc_oos", "plus")
        labels = all_dataset["train"].features["intent"].names
        labels = [" " + l.replace("_", " ") for l in labels]

        all_dataset = all_dataset.map(lambda example: {**example, "answer": labels[example["intent"]], "options": labels})
        train_dataset = all_dataset["train"]
        test_dataset = all_dataset["validation"]

        template = "{instruction}Utterance: {text}\nIntent:{answer}"
        domain_prompt = "Intent:"
        recalibrate_every = False
        balanced_sampling = True
        instruction = "Instruction: Choose the correct intent for the utterance.\n"

    else:
        raise NotImplementedError

    return {
        "train": train_dataset,
        "test": test_dataset,
        "template": template,
        "document_template": document_template,
        "recalibrate_every": recalibrate_every,
        "balanced_sampling": balanced_sampling,
        "domain_prompt": domain_prompt,
        "instruction": instruction,
        "truncate_seperator": truncate_seperator,
        "use_rouge": use_rouge,
    }

DATASET_TO_TASK = {
    # Open-domain QA
    "nq": "generate",
    "popqa": "generate",
    "triviaqa": "generate",

    # Scrolls/ZeroScrolls
    "qmsum": "generate",
    "summ_screen_fd": "generate",
    "govreport": "generate",
    "scrolls/govreport": "generate",

    "narrativeqa": "generate",
    "qasper": "generate",
    "quality": "generate",

    # datasets from PCW
    "sst2": "loglikelihood",
    "ag_news": "loglikelihood",
    "mr": "loglikelihood",
    "sst5": "loglikelihood",
    "nlu_scenario": "loglikelihood",
    "trec-coarse": "loglikelihood",
    "trec-fine": "loglikelihood",
    "dbpedia": "loglikelihood",
    "nlu_intent": "loglikelihood",
    "banking77": "loglikelihood",
    "clinic150": "loglikelihood",
}

def load_lm_dataset(dataset):
    if dataset == "pg19":
        eval_dataset = load_dataset("emozilla/pg19-test")["test"]
        text_column_name = "text" 

    elif dataset == "proofpile":
        eval_dataset = load_dataset("hoskinson-center/proof-pile")["test"]
        text_column_name = "text" 

    elif dataset == "codeparrot":
        eval_dataset = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup")["train"]
        text_column_name = "content" 

    else:
        raise NotImplementedError
        
    return eval_dataset, text_column_name


def add_contriever_scores(dataset, llama_tokenizer):
    import torch
    from transformers import AutoModel, AutoTokenizer

    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    contriever_model = AutoModel.from_pretrained("facebook/contriever",)
    contriever_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contriever_model = contriever_model.to(device)

    def mean_pooling(token_embeddings, mask ):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def calculate_score(query, passages):
        # calculate the score between query and passages
        query_text = llama_tokenizer.decode(query)
        query = contriever_tokenizer(query_text, return_tensors="pt", max_length=512, truncation=True)
        query = {k: v.cuda() for k, v in query.items()}
        q_embed = contriever_model(**query)
        q_embed = mean_pooling(q_embed[0], query["attention_mask"])

        passages_text = llama_tokenizer.batch_decode(passages)
        passages = contriever_tokenizer(passages_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        passages = {k: v.cuda() for k, v in passages.items()}
        p_embed = contriever_model(**passages)
        p_embed = mean_pooling(p_embed[0], passages["attention_mask"])

        score = torch.inner(q_embed, p_embed)
        return score

    def add_scores(example):
        query = example["input_ids"][:256]
        passages = example["encoder_input_ids"]
        with torch.inference_mode():
            scores = calculate_score(query, passages)
            example["scores"] = scores.cpu().numpy()
        return example
    
    dataset = dataset.map(add_scores, batched=False, )
    return dataset