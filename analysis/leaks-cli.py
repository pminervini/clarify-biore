#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import pickle
import logging

from typing import List, Dict, Set, Tuple, Any

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def jsonl_to_lst(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        res = [json.loads(line) for line in f]
    return res


def to_cui_triples(entry_lst: List[Dict[str, Any]], name_to_cuis: Dict[str, Set[str]]) -> Set[Tuple[str, str, str]]:
    logger.info('Generating training CUI triples ..')

    cui_triples_set: Set[Tuple[str, str, str]] = set()
    for entry in entry_lst:
        s, o = entry['group']
        s_cuis, r, o_cuis = name_to_cuis[s], entry['relation'], name_to_cuis[o]
        cui_triples_set |= {(s_cui, r, o_cui) for s_cui in s_cuis for o_cui in o_cuis}

    return cui_triples_set


def main(argv):
    logger.info('Loading data/processed/complete_{train/dev/test}.txt ..')
    train_lst: List[Dict[str, Any]] = jsonl_to_lst('data/processed/complete_train.txt')
    dev_lst: List[Dict[str, Any]] = jsonl_to_lst('data/processed/complete_dev.txt')
    test_lst: List[Dict[str, Any]] = jsonl_to_lst('data/processed/complete_test.txt')

    # logger.info('Loading data/UMLS/processed/umls.cui_to_txts.pkl ..')
    # with open('data/UMLS/processed/umls.cui_to_txts.pkl', 'rb') as f:
    #     cui_to_names: Dict[str, Set[str]] = pickle.load(f)

    logger.info('Loading data/UMLS/processed/umls.txt_to_cui.pkl ..')
    with open('data/UMLS/processed/umls.txt_to_cui.pkl', 'rb') as f:
        name_to_cuis: Dict[str, Set[str]] = pickle.load(f)

    logger.info('Loading data/UMLS/processed/umls.reltxt_to_groups.pkl ..')
    with open('data/UMLS/processed/umls.reltxt_to_groups.pkl', 'rb') as f:
        reltxt_to_groups: Dict[str, Set[Tuple[str, str]]] = pickle.load(f)

    # logger.info('Generating CUI triples ..')
    # cui_triples_set: Set[Tuple[str, str, str]] = {(s, r, o) for r, gs in reltxt_to_groups.items() for s, o in gs}

    logger.info('Generating training CUI triples ..')
    train_cui_triples: Set[Tuple[str, str, str]] = to_cui_triples(train_lst, name_to_cuis)
    train_cui_triples_inv: Set[Tuple[str, str, str]] = {(o, r, s) for (s, r, o) in train_cui_triples}

    train_cui_pairs = {(s, o) for (s, r, o) in train_cui_triples}
    train_cui_pairs_inv = {(o, s) for (s, r, o) in train_cui_triples}

    for name, eval_lst in [('Dev', dev_lst), ('Test', test_lst)]:
        eval_cui_triples = to_cui_triples(eval_lst, name_to_cuis)

        eval_cui_pairs = {(s, o) for (s, r, o) in eval_cui_triples}
        eval_cui_pairs_inv = {(o, s) for (s, r, o) in eval_cui_triples}

        logger.info(f'Training set of CUI triples -- size: {len(train_cui_triples)}')
        logger.info(f'{name} set of CUI triples -- size: {len(eval_cui_triples)}')

        inter_triples = train_cui_triples & eval_cui_triples
        union_triples = train_cui_triples | eval_cui_triples

        inter_triples_inv = train_cui_triples_inv & eval_cui_triples
        union_triples_inv = train_cui_triples_inv | eval_cui_triples

        inter_pairs = train_cui_pairs & eval_cui_pairs
        union_pairs = train_cui_pairs | eval_cui_pairs

        inter_pairs_inv = train_cui_pairs_inv & eval_cui_pairs
        union_pairs_inv = train_cui_pairs_inv | eval_cui_pairs

        logger.info(f'Training/{name} intersection size: {len(inter_triples)}')
        logger.info(f'Training/{name} union size: {len(union_triples)}')

        logger.info(f'Number of {name} triples in Training: '
                    f'{(len(inter_triples) / len(eval_cui_triples)) * 100:.2f}%')

        logger.info(f'Inverse Training/{name} intersection size: {len(inter_triples_inv)}')
        logger.info(f'Inverse Training/{name} union size: {len(union_triples_inv)}')

        logger.info(f'Number of {name} triples in Inverse Training: '
                    f'{(len(inter_triples_inv) / len(eval_cui_triples)) * 100:.2f}%')

        logger.info(f'Number of {name} triples in Training or Inverse Training: '
                    f'{(len(inter_triples | inter_triples_inv) / len(eval_cui_triples)) * 100:.2f}%')

        logger.info(f'Number of {name} pairs in Training: '
                    f'{(len(inter_pairs) / len(eval_cui_pairs)) * 100:.2f}%')

        logger.info(f'Number of {name} pairs in Inverse Training: '
                    f'{(len(inter_pairs_inv) / len(eval_cui_pairs)) * 100:.2f}%')


if __name__ == "__main__":
    main(sys.argv[1:])
