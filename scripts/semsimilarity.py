# -*- coding: utf-8 -*-
"""
    Compare sentence similarities between input sentence and three
    contrastive option including antonym substitution from the source.
    E.g., He was asleep. -- He was not asleep. -- He was awake. -- He was not awake.

    Calculates accuracy on the negated antonym test suite.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Get sentence similarities.")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or directory, e.g., bert-base-cased",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Set `t5` for t5 type models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data for semantic similarity task (jsonlines)."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        help="Pooling operation (cls, mean, max)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="./tmp",
        help="Temporary directory for downloaded models, data, etc."
    )

    args = parser.parse_args()

    return args


def read_data(data_f):
    data = []
    with open(data_f, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def encode_sentences(sentences, tokenizer):
    tokens = {"input_ids": [], "attention_mask": []}

    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True,
                                           padding="max_length",
                                           return_tensors="pt")

        new_tokens.to(DEVICE)

        tokens["input_ids"].append(new_tokens["input_ids"][0])
        tokens["attention_mask"].append(new_tokens["attention_mask"][0])

    tokens["input_ids"] = torch.stack(tokens["input_ids"])
    tokens["attention_mask"] = torch.stack(tokens["attention_mask"])

    return tokens


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value

    return torch.max(token_embeddings, 1)[0]


def cls_pooling(model_output, attention_mask):

    return model_output.last_hidden_state[:,0]


def calculate_accuracy(predictions, labels):
    correct = 0.0
    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1

    acc = 1.0 * correct / len(predictions)

    return acc


def main():
    args = parse_args()

    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)

    model.to(DEVICE)

    sentences = read_data(args.data)

    preds = []
    labels = []

    if args.model_type is not None:
        if args.model_type == "t5":
            for d in sentences:
                labels.append(d["label"])
                src_sentences_enc = tokenizer(d["input"], padding=True, truncation=True, return_tensors="pt")
                options_enc = tokenizer(d["sentences"], padding=True, truncation=True, return_tensors="pt")

                src_output = model.encoder(input_ids=src_sentences_enc["input_ids"],
                                   attention_mask=src_sentences_enc["attention_mask"],
                                   return_dict=True)

                options_output = model.encoder(input_ids=options_enc["input_ids"],
                                       attention_mask=options_enc["attention_mask"],
                                       return_dict=True)

                src_embeddings = src_output.last_hidden_state
                options_embeddings = options_output.last_hidden_state

                pooled_src_embeddings = mean_pooling(src_embeddings, src_sentences_enc["attention_mask"])
                pooled_options_embeddings = mean_pooling(options_embeddings, options_enc["attention_mask"])

                normalized_src = F.normalize(pooled_src_embeddings, p=2, dim=1)
                normalized_opt = F.normalize(pooled_options_embeddings, p=2, dim=1)

                pooled_src = normalized_src.detach().cpu().numpy()
                pooled_option = normalized_opt.detach().cpu().numpy()

                sim = cosine_similarity(pooled_src.reshape(1, -1), pooled_option[0:])

                preds.append(np.argmax(sim))

            accuracy = calculate_accuracy(preds, labels)

    else:
        for d in sentences:
            labels.append(d["label"])
            src_sentences = encode_sentences([d["input"]], tokenizer)
            options = encode_sentences(d["sentences"], tokenizer)

            with torch.no_grad():
                src_outputs = model(**src_sentences)
                option_outputs = model(**options)

            if args.pooling == "mean":
                src_embeddings = src_outputs.last_hidden_state
                option_embeddings = option_outputs.last_hidden_state

                src_attention_mask = src_sentences["attention_mask"]
                option_attention_mask = options["attention_mask"]

                pooled_src_embeddings = mean_pooling(src_embeddings, src_attention_mask)
                pooled_option_embeddings = mean_pooling(option_embeddings, option_attention_mask)

            if args.pooling == "max":
                src_embeddings = src_outputs
                option_embeddings = option_outputs

                src_attention_mask = src_sentences["attention_mask"]
                option_attention_mask = options["attention_mask"]

                pooled_src_embeddings = max_pooling(src_embeddings, src_attention_mask)
                pooled_option_embeddings = max_pooling(option_embeddings, option_attention_mask)

            if args.pooling == "cls":
                src_embeddings = src_outputs
                option_embeddings = option_outputs

                pooled_src_embeddings = cls_pooling(src_embeddings)
                pooled_option_embeddings = cls_pooling(option_embeddings)

            normalized_src = F.normalize(pooled_src_embeddings, p=2, dim=1)
            normalized_opt = F.normalize(pooled_option_embeddings, p=2, dim=1)

            pooled_src = normalized_src.detach().cpu().numpy()
            pooled_option = normalized_opt.detach().cpu().numpy()

            sim = cosine_similarity(pooled_src.reshape(1, -1), pooled_option[0:])

            preds.append(np.argmax(sim))

        accuracy = calculate_accuracy(preds, labels)

    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    print(f"Accuracy: {'{0:.3g}'.format(accuracy)}")

    model_name = args.model.split("/")[-1]

    out_file = f"./results/results-{model_name}.txt"
    pred_file = f"./results/misclassified-by-{model_name}.txt"

    with open(out_file, "w+") as f:
        f.write(f"Sentence similarity on {args.data} data.\n")
        f.write(f"Model: {args.model}.\n")
        f.write(f"Accuracy: {'{0:.3g}'.format(accuracy)}\n")
        f.write(f"Misclassified examples: \n")

        for i, (p, l) in enumerate(zip(preds, labels)):
            if p != l:
                f.write(f"{sentences[i]}" + "\t" + f"Predicted label: {p}." + "\n")

    with open(pred_file, "w+") as f:
        for i, (p, l) in enumerate(zip(preds, labels)):
            if p != l:
                f.write(f"{sentences[i]}" + "\n")


if __name__ == "__main__":
    main()
