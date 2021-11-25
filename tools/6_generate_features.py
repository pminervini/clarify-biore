#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import torch
from torch import Tensor
from transformers import BertTokenizer

import config
from utils.utils import JsonlReader, read_entities, read_relations

from typing import List, Dict, Tuple, Any

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_jsonl(jsonl: Dict[str, Any], tokenizer, entity2idx: Dict[str, int], relation2idx: Dict[str, int], idx,
                   max_seq_length: int = 128, e1_tok: str = "$", e2_tok: str = "^") -> Any:

    # jsonl looks like this:

    # {
    #     "group": [
    #         "screw loosening",
    #         "plate fracture"
    #     ],
    #     "relation": "temporally_followed_by",
    #     "sentences": [
    #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
    #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    #         [..]
    #     ],
    #     "e1": null,
    #     "e2": null,
    #     "reldir": 0
    # }

    # Group
    src, tgt = jsonl["group"]
    relation = jsonl["relation"]

    input_ids: List[Tensor] = list()
    entity_ids: List[Tensor] = list()
    attention_mask: List[Tensor] = list()

    # For each sentence ...
    for sent in jsonl["sentences"]:
        # Tokenize it
        encoded = tokenizer.encode_plus(sent, max_length=max_seq_length, pad_to_max_length=True, return_tensors='pt')

        input_ids_i = encoded["input_ids"]
        attention_mask_i = encoded["attention_mask"]
        entity_ids_i = torch.zeros(max_seq_length)

        # Can happen for long sentences when entity markers go out of boundary, ignore such cases
        try:
            e1_start, e1_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e1_tok]).flatten()
        except:
            return [], []

        entity_ids_i[e1_start + 1:e1_end] = 1

        try:
            e2_start, e2_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e2_tok]).flatten()
        except:
            return [], []

        entity_ids_i[e2_start + 1:e2_end] = 2
        input_ids.append(input_ids_i)
        entity_ids.append(entity_ids_i.unsqueeze(0))
        attention_mask.append(attention_mask_i)

    if len(input_ids) == 0: # Should not happen
        return []

    group = (entity2idx[src], entity2idx[tgt])

    features = [dict(
        input_ids=torch.cat(input_ids),
        entity_ids=torch.cat(entity_ids),
        attention_mask=torch.cat(attention_mask),
        label=relation2idx[relation.lower()],
        group=group,
    ), ]
    return features


def load_tokenizer(model_dir: str, do_lower_case: bool = False):
    return BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)


def create_features(jsonl_fname: str, tokenizer, output_fname: str, entity2idx: Dict[str, int], relation2idx: Dict[str, int],
                    max_seq_length: int = 128, e1_tok: str = "$", e2_tok: str = "^"):
    jr = list(iter(JsonlReader(jsonl_fname)))

    # Each entry in jr looks like the following:

    # {
    #     "group": [
    #         "screw loosening",
    #         "plate fracture"
    #     ],
    #     "relation": "temporally_followed_by",
    #     "sentences": [
    #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
    #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    #         [..]
    #     ],
    #     "e1": null,
    #     "e2": null,
    #     "reldir": 0
    # }

    features = list()
    logger.info("Loading {} lines from complete data txt file.".format(len(jr)))
    for idx, jsonl in enumerate(jr):
        if idx % 10000 == 0 and idx != 0:
            logger.info("Created {} features".format(idx))

        feats = tokenize_jsonl(jsonl, tokenizer, entity2idx, relation2idx, idx, max_seq_length, e1_tok, e2_tok)

        # feats looks like this:
        #   input_ids: torch.cat(input_ids),
        #   entity_ids: torch.cat(entity_ids),
        #   attention_mask: torch.cat(attention_mask),
        #   label: relation2idx[relation.lower()],
        #   group: group,

        features.extend(feats)
    torch.save(features, output_fname)
    logger.info("Saved {} lines of features.".format(len(features)))


if __name__ == "__main__":
    # Loads a tokenizer via BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)
    tokenizer = load_tokenizer(config.pretrained_model_dir, config.do_lower_case)

    # Read all entities and relations, and assign indices to them
    entity2idx = read_entities(config.entities_file)
    relation2idx = read_relations(config.relations_file)

    logger.info("Read {} unique entities and {} unique relations.".format(len(entity2idx), len(relation2idx)))

    # Compute the features for complete_train.txt, complete_dev.txt, etc.
    files = [
        # ./data/processed/complete_train.txt, ./data/processed/features/train.pt
        (config.complete_train, config.feats_file_train),
        # ./data/processed/complete_dev.txt, ./data/processed/features/dev.pt
        (config.complete_dev, config.feats_file_dev),
        # ./data/processed/complete_test.txt, ./data/processed/features/test.pt
        (config.complete_test, config.feats_file_test)
    ]

    for input_fname, output_fname in files:
        logger.info("Creating features for input `{}` ...".format(input_fname))
        # Generate the features from e.g. complete_train.txt, where each line looks like the following:

        # {
        #     "group": [
        #         "screw loosening",
        #         "plate fracture"
        #     ],
        #     "relation": "temporally_followed_by",
        #     "sentences": [
        #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
        #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
        #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
        #         [..]
        #     ],
        #     "e1": null,
        #     "e2": null,
        #     "reldir": 0
        # }

        create_features(input_fname, tokenizer, output_fname, entity2idx, relation2idx, config.max_seq_length)

        # Features look like the following:

        # features = [{
        #     input_ids: torch.cat(input_ids),
        #     entity_ids: torch.cat(entity_ids),
        #     attention_mask: torch.cat(attention_mask),
        #     label: relation2idx[relation.lower()],
        #     group: group,
        # }, [..] ]

        # They are saved in e.g. ./data/processed/features/train.pt

        logger.info("Saved features at `{}` ...".format(output_fname))
