#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import torch
from transformers import BertTokenizer

import config
from utils.utils import JsonlReader, read_entities, read_relations

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_jsonl(jsonl, tokenizer, entity2idx, relation2idx, idx, max_seq_length=128, e1_tok="$", e2_tok="^"):
    # Group
    src, tgt = jsonl["group"]
    relation = jsonl["relation"].lower()
    ent_names = jsonl["ent_names"]
    input_ids = list()
    entity_ids = list()
    attention_mask = list()

    # jsonl looks like this:

    # {
    #     "group": [
    #         "pathologic function",
    #         "therapeutic or preventive procedure"
    #     ],
    #     "relation": "temporally_followed_by",
    #     "ent_names": [
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [..]
    #     ],
    #     "sentences": [
    #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
    #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    #         "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
    #         [..]
    #     ]
    # }

    for sent in jsonl["sentences"]:
        # Tokenize it
        encoded = tokenizer.encode_plus(sent, max_length=max_seq_length, pad_to_max_length=True, return_tensors='pt')

        input_ids_i = encoded["input_ids"]
        attention_mask_i = encoded["attention_mask"]
        entity_ids_i = torch.zeros(max_seq_length)

        # Can happen for long sentences when entity markers go out of boundary, ignore such cases
        e1_start, e1_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e1_tok]).flatten()

        entity_ids_i[e1_start + 1:e1_end] = 1

        e2_start, e2_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e2_tok]).flatten()

        entity_ids_i[e2_start + 1:e2_end] = 2
        input_ids.append(input_ids_i)
        entity_ids.append(entity_ids_i.unsqueeze(0))
        attention_mask.append(attention_mask_i)

    # Happened once -- somehow?!
    if len(input_ids) == 0:
        return []

    group = (entity2idx[src], entity2idx[tgt])

    features = [dict(
        input_ids=torch.cat(input_ids),
        entity_ids=torch.cat(entity_ids),
        attention_mask=torch.cat(attention_mask),
        label=relation2idx[relation],
        group=group,
        ent_names=ent_names
    ), ]

    return features


def load_tokenizer(model_dir, do_lower_case=False):
    return BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)


def create_features(jsonl_fname, tokenizer, output_fname, entity2idx, relation2idx,
                    max_seq_length=128, e1_tok="$", e2_tok="^"):
    jr = list(iter(JsonlReader(jsonl_fname)))

    # Each line in jr, which is e.g. ./data/processed/complete_types_train.txt, looks like the following:

    # {
    #     "group": [
    #         "pathologic function",
    #         "therapeutic or preventive procedure"
    #     ],
    #     "relation": "temporally_followed_by",
    #     "ent_names": [
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [
    #             "screw loosening",
    #             "plate fracture"
    #         ],
    #         [..]
    #     ],
    #     "sentences": [
    #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
    #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    #         "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
    #         [..]
    #     ]
    # }

    features = list(), list()
    logger.info("Loading {} lines from complete data txt file.".format(len(jr)))
    for idx, jsonl in enumerate(jr):
        if idx % 10000 == 0 and idx != 0:
            logger.info("Created {} features".format(idx))

        # Featurizes jsonl
        feats = tokenize_jsonl(jsonl, tokenizer, entity2idx, relation2idx, idx, max_seq_length, e1_tok, e2_tok)

        # Adds the featurised jsonl to features
        features.extend(feats)
    torch.save(features, output_fname)
    logger.info("Saved {} lines of features.".format(len(features)))


if __name__ == "__main__":
    # Loads a tokenizer via BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)
    tokenizer = load_tokenizer(config.pretrained_model_dir, config.do_lower_case)

    # Read all entities and relations, and assign indices to them
    entity2idx = read_entities(config.entities_file_types)
    relation2idx = read_relations(config.relations_file_types)

    logger.info("Read {} unique entities and {} unique relations.".format(len(entity2idx), len(relation2idx)))

    files = [
        # ./data/processed/complete_types_train.txt, ./data/processed/features/types_train.pt
        (config.complete_types_train, config.feats_file_types_train),
        (config.complete_types_dev, config.feats_file_types_dev),
        (config.complete_types_test, config.feats_file_types_test)
    ]

    for input_fname, output_fname in files:
        logger.info("Creating features for input `{}` ...".format(input_fname))
        # Generate the features from e.g. ./data/processed/complete_types_train.txt, where each line looks like:

        # {
        #     "group": [
        #         "pathologic function",
        #         "therapeutic or preventive procedure"
        #     ],
        #     "relation": "temporally_followed_by",
        #     "ent_names": [
        #         [
        #             "screw loosening",
        #             "plate fracture"
        #         ],
        #         [
        #             "screw loosening",
        #             "plate fracture"
        #         ],
        #         [
        #             "screw loosening",
        #             "plate fracture"
        #         ],
        #         [
        #             "screw loosening",
        #             "plate fracture"
        #         ],
        #         [..]
        #     ],
        #     "sentences": [
        #         "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
        #         "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
        #         "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
        #         "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
        #         [..]
        #     ]
        # }

        # This generates e.g. ./data/processed/features/types_train.pt
        create_features(input_fname, tokenizer, output_fname, entity2idx, relation2idx, config.max_seq_length)

        # Features look like the following:

        # features = [{
        #     input_ids: torch.cat(input_ids),
        #     entity_ids: torch.cat(entity_ids),
        #     attention_mask: torch.cat(attention_mask),
        #     label: relation2idx[relation.lower()],
        #     group: group,
        # }, [..] ]

        # They are saved in e.g. ./data/processed/features/types_train.pt

        logger.info("Saved features at `{}` ...".format(output_fname))
