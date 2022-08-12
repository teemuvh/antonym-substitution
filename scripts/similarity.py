# -*- coding: utf-8 -*-
"""
    Compare sentence similarities between input sentence and three
    contrastive option including antonym substitution from the source.
    E.g., He was asleep. -- He was not asleep. -- He was awake. -- He was not awake.
"""

import argparse
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Get sentence similarities.")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or directory, e.g., bert-base-cased",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data for semantic similarity task (jsonlines)."
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

    sentences = read_data(args.data)

    preds = []
    labels = []

    for d in sentences:
        labels.append(d["label"])
        src_sentences = encode_sentences([d["input"]], tokenizer)
        options = encode_sentences(d["sentences"], tokenizer)

        src_outputs = model(**src_sentences)
        option_outputs = model(**options)

        src_embeddings = src_outputs.last_hidden_state
        option_embeddings = option_outputs.last_hidden_state

        src_attention_mask = src_sentences["attention_mask"]
        option_attention_mask = options["attention_mask"]

        mean_pooled_src_embeddings = mean_pooling(src_embeddings, src_attention_mask)
        mean_pooled_option_embeddings = mean_pooling(option_embeddings, option_attention_mask)

        mean_pooled_src = mean_pooled_src_embeddings.detach().numpy()
        mean_pooled_option = mean_pooled_option_embeddings.detach().numpy()

        sim = cosine_similarity(mean_pooled_src.reshape(1, -1), mean_pooled_option[0:])
        preds.append(np.argmax(sim))

    accuracy = calculate_accuracy(preds, labels)
    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    print(f"Accuracy: {'{0:.3g}'.format(accuracy)}")


if __name__ == "__main__":
    main()
