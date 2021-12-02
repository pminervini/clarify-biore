#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import itertools
import json
import logging
import os
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split

import config
from utils.utils import JsonlReader

from typing import Dict, List, Set, Tuple, Iterable, Any

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(config.SEED)
random.seed(config.SEED)


def get_groups_texts_from_umls_vocab(relation_text_to_groups: Dict[str, Set[Tuple[str, str]]],
                                     cui_to_entity_texts: Dict[str, Set[str]],
                                     fname_reltext_all_combos: str,
                                     load_existing: bool = True) -> Set[str]:
    # Load rel groups file, if it exists
    if os.path.isfile(fname_reltext_all_combos) and load_existing:
        logger.info('Found existing relation text combination file -- loading relations.')
        with open(fname_reltext_all_combos, 'rb') as f:
            return pickle.load(f)
    else:
        logger.info("Not loading relation text combination file. Generating new set:")

    # Get all related CUI groups into set:
    groups: Set[Tuple[str, str]] = set()
    for relation_text in relation_text_to_groups:
        groups.update(relation_text_to_groups[relation_text])

    # Collect all combinations of related entities
    logger.info("Collecting all possible textual combinations of CUI groups ...")
    groups_texts: Set[str] = set()
    l = len(groups)

    for idx, (cui_src, cui_tgt) in enumerate(groups):
        if idx % 100000 == 0 and idx != 0:
            logger.info("Parsed {} groups of {}".format(idx, l))
        cui_src_texts = cui_to_entity_texts[cui_src]
        cui_tgt_texts = cui_to_entity_texts[cui_tgt]
        for cui_src_text_i in cui_src_texts:
            temp = list(zip([cui_src_text_i] * len(cui_tgt_texts), cui_tgt_texts))
            temp = ["\t".join(i) for i in temp]
            groups_texts.update(temp)

    # NOTE: this consumes a LOT of memory (~18 GB)! (clearing up memory takes around half an hour)
    logger.info("Collected {} unique tuples of (src_entity_text, tgt_entity_text) type.".format(len(groups_texts)))

    # Save rel groups:
    with open(fname_reltext_all_combos, 'wb') as f:
        logger.info('Saving relation text combination file.')
        pickle.dump(groups_texts, f)

    return groups_texts


def align_groups_to_sentences(groups_texts: Set[str],  # Set of tab-separated sentence pairs
                              jsonl_fname: str,  # Input file: umls.linked_sentences.jsonl
                              output_fname: str) -> Tuple[Set[str], Set[str]]:  # Output file: umls.linked_sentences_to_groups.jsonl
    jr = JsonlReader(jsonl_fname)

    with open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
        logger.info("Aligning texts (sentences) to groups ...")
        pos_groups: Set[str] = set()
        neg_groups: Set[str] = set()

        for idx, jdata in enumerate(jr):
            if idx % 1000000 == 0 and idx != 0:
                logger.info("Processed {} tagged sentences".format(idx))

            # Permutations of size for matched entities in a sentence
            matched_perms = set(itertools.permutations(jdata['matches'].keys(), 2))

            # Left-hand-side (lhs) <==> right-hand-side (rhs)
            lhs2rhs = collections.defaultdict(list)
            rhs2lhs = collections.defaultdict(list)

            for group in matched_perms:
                src, tgt = group
                lhs2rhs[src].append(tgt)
                rhs2lhs[tgt].append(src)

            # Since `groups_texts` contain all possible groups that can exist
            # in the UMLS KG, for some relation, the intersection of this set
            # with matched permuted groups efficiently yields groups which
            # **do exist in KG for some relation and have matching sentences**.
            matched_perms = {"\t".join(m) for m in matched_perms}
            common = groups_texts.intersection(matched_perms)

            # We use sentence level noise, i.e., for the given sentence the
            # common groups represent positive groups, while the negative
            # samples can be generated as follows (like open-world assumption):
            #
            # For a +ve group, with prob. 1/2, remove the left (src) or right
            # (tgt) entity and replace with N entities such that the negative
            # group (e_orig, e_replaced) [for rhs] / (e_replaced, e_orig) [for lhs]
            # **must not be in KG for any relation**. This technique can possibly be
            # seen as creating hard negatives for same text evidence.
            output = {"p": set(), "n": set()}

            for group in common:
                pos_groups.add(group)
                src, tgt = group.split("\t")
                output["p"].add(group)
                # Choose left or right side to corrupt
                lhs_or_rhs = random.choice([0, 1])

                if lhs_or_rhs == 0:
                    for corrupt_tgt in lhs2rhs[src]:
                        negative_group = "{}\t{}".format(src, corrupt_tgt)
                        if negative_group not in common:
                            output["n"].add(negative_group)
                else:
                    for corrupt_src in rhs2lhs[tgt]:
                        negative_group = "{}\t{}".format(corrupt_src, tgt)
                        if negative_group not in common:
                            output["n"].add(negative_group)

            if output["p"] and output["n"]:
                no = list(output["n"])
                random.shuffle(no)
                # Keep number of negative groups at most as positives
                no = no[:len(output["p"])]
                output["n"] = no
                output["p"] = list(output["p"])
                neg_groups.update(no)
                jdata["groups"] = output
                wf.write(json.dumps(jdata) + "\n")

    # There will be lot of negative groups, so we will remove them next!
    logger.info(f"Collected {len(pos_groups)} positive and {len(neg_groups)} negative groups.")

    return pos_groups, neg_groups


def pruned_triples(cui_to_entity_texts: Dict[str, Set[str]],
                   relation_text_to_groups: Dict[str, Set[Tuple[str, str]]],
                   pos_groups: Set[str],
                   neg_groups: Set[str],
                   min_rel_group: int = 10,
                   max_rel_group: int = 1500) -> List[Tuple[str, str, str]]:
    logger.info("Mapping CUI groups to relations ...")
    group_to_relation_texts = collections.defaultdict(list)

    # relation_text_to_groups:
    #   Keys: 'has_component', 'has_measured_component', 'measures', ...
    #   Values of relation_text_to_groups['has_component']: {('C0066621', 'C1953785'), ('C0883230', 'C0942242'), ...}

    for idx, (relation_text, groups) in enumerate(relation_text_to_groups.items()):
        for group in groups:
            group_to_relation_texts[group].append(relation_text)  # can have multiple rel texts per group

    # group_to_relation_texts:
    #   Keys: ('C0066621', 'C1953785'), ('C0883230', 'C0942242'), ...
    #   Values: ['has_component', 'has_measured_component', 'measures', ...]

    logger.info("Mapping relations to groups texts ...")
    relation_text_to_groups_texts = collections.defaultdict(set)

    for idx, (group, relation_texts) in enumerate(group_to_relation_texts.items()):
        if idx % 1000000 == 0 and idx != 0:
            logger.info(f"Mapped from {idx} groups")

        cui_src, cui_tgt = group
        local_groups = set()

        # Sets of mentions of the entities cui_src and cui_tgt
        cui_src_texts: Set[str] = cui_to_entity_texts[cui_src]
        cui_tgt_texts: Set[str] = cui_to_entity_texts[cui_tgt]

        for l1i in cui_src_texts:
            local_groups.update(list(zip([l1i] * len(cui_tgt_texts), cui_tgt_texts)))

        # local_groups is a set that contains all pairs of entity mentions (e1, e2), as tuples, where:
        #   - e1 is from cui_src_texts
        #   - e2 is from cui_tgt_texts

        # Then for each pair of mentions ..
        for lg in local_groups:
            # If the pair is in pos_groups ..
            if "\t".join(lg) in pos_groups:
                # For each relation type linking these two entities ..
                for relation_text in relation_texts:
                    # Add the tab-separated entity mention pairs to relation_text_to_groups_texts[relation_text]
                    relation_text_to_groups_texts[relation_text].add("\t".join(lg))

    # relation_text_to_groups_texts maps one relation names with the entity mentions that also appear in pos_groups
    logger.info("No. of relations before pruning: {}".format(len(relation_text_to_groups_texts)))

    # Prune relations based on the group size
    relations_to_del = list()
    for relation_text, groups_texts in relation_text_to_groups_texts.items():
        if (len(groups_texts) < min_rel_group) or (len(groups_texts) > max_rel_group):
            relations_to_del.append(relation_text)

    logger.info("Relations not matching the criterion of min, max group sizes of {} and {}.".format(min_rel_group,
                                                                                                    max_rel_group))
    # Delete relations not meeting min and max counts
    for r in relations_to_del:
        del relation_text_to_groups_texts[r]
    logger.info("Relations deleted: {}".format(relations_to_del))
    logger.info("No. of relations after pruning: {}".format(len(relation_text_to_groups_texts)))

    # Update positive groups
    new_pos_groups = set()
    entities = set()

    # For each pair relation name R, tab-separated pair of entity mentions related by R ..
    for relation_text, groups_texts in relation_text_to_groups_texts.items():
        # For each tab-separated pair of entity mentions ..
        for group_text in groups_texts:
            # Add them to the set group_text, and add the mentions to the set of entities.
            new_pos_groups.add(group_text)
            entities.update(group_text.split("\t"))

    logger.info(f"Updated no. of positive groups after pruning: {len(new_pos_groups)}")
    logger.info(f"No. of unique entities: {len(entities)}")

    # Update negative groups

    # 1) We apply the constraint that the negative groups must have positive
    # triples entities only
    new_neg_groups = set()

    # Then, new_neg_groups will contain all entity pairs originally contained in neg_groups involving known entities.
    for negative_group in neg_groups:
        src, tgt = negative_group.split("\t")
        if (src in entities) and (tgt in entities):
            new_neg_groups.add(negative_group)

    logger.info(f"[1] Updated no. of negative groups after pruning groups that are not in positive entities: {len(new_neg_groups)}")

    # 2) Negative examples are used for NA / Other relation, which is just another class.
    # To avoid training too much on NA relation, we make a simple choice randomly taking
    # the same number of groups as largest group size positive class.
    max_pos_group_size = max([len(v) for v in relation_text_to_groups_texts.values()])
    new_neg_groups = list(new_neg_groups)
    random.shuffle(new_neg_groups)

    # Using 70% of positive groups to form negative groups
    new_neg_groups = new_neg_groups[:int(max_pos_group_size * 0.7)]
    logger.info(f'Len of new_pos_groups: {len(new_pos_groups)}, Len of max_pos_group_size: {max_pos_group_size}')
    logger.info(f"Number of negative groups after taking 70 percent more than positive groups: {len(new_neg_groups)}")

    # Here relation_text_to_groups_texts['NA'] will correspond to a set of negative tab-separate entity mentions
    relation_text_to_groups_texts["NA"] = new_neg_groups  # new_neg_groups here is a list but indeed should be a set

    # Collect triples now
    triples = set()

    # For each pair relation name R, set of tab-separated entity mentions (e1, e2) linked by R ..
    for r, groups in relation_text_to_groups_texts.items():
        # For each pair (e1, e2) in the set ..
        for group in groups:
            # Split the two entity mentions ..
            src, tgt = group.split("\t")
            # And add the corresponding triple to the set of triples.
            triples.add((src, r, tgt))
    triples = list(triples)

    logger.info(f" *** No. of triples (including NA) *** : {len(triples)}")

    return triples


def filter_triples_with_evidence(triples: List[Tuple[str, str, str]],
                                 max_bag_size: int = 32) -> Tuple[Set[Tuple[str, str, str]],
                                                                  Dict[str, Dict[str, Iterable[str]]]]:
    group_to_relation_texts: Dict[str, Set[str]] = collections.defaultdict(set)

    # group_to_relation_texts contains:
    #   - Keys: tab-separated entity mention pairs
    #   - Values: each value is a set of relation names
    for ei, rj, ek in triples:
        group = "{}\t{}".format(ei, ek)
        group_to_relation_texts[group].add(rj)

    jr = JsonlReader(config.groups_linked_sents_file)

    # Each entry in config.groups_linked_sents_file (default: ./data/MEDLINE/processed/umls.linked_sentences_to_groups.jsonl)
    # is a Dict that contains the following entries:
    # - "sent": "To investigate the medium term outcome of cardiac surgery, we evaluated patients over ..."
    # - "matches": { "medium": [19, 25], "term": [26, 30], "outcome": [31, 38], ...}
    # - "groups": {
    #     "p": ["age\tpatients", "patients\tage"],
    #     "n": ["patients\t1.5", "patients\tyears"]
    #   }

    group_to_data: Dict[str, Dict[int, Iterable[str]]] = collections.defaultdict(list)

    # Initialised as a set of tab-separated entity mention pairs, related (R != 'NA') or not (R = 'NA'), from the Dict
    # that maps entity mention pairs (groups) to relation names.
    candid_groups = set(group_to_relation_texts.keys())

    # We iterate through each record in ./data/MEDLINE/processed/umls.linked_sentences_to_groups.jsonl ...
    for idx, jdata in enumerate(jr):
        if idx % 1000000 == 0 and idx != 0:
            logger.info("Processed {} lines for linking to triples".format(idx))

        # We identify in the positive and negative groups the entity mention pairs that overlap with candid_groups ...
        common = candid_groups.intersection(jdata["groups"]["p"] + jdata["groups"]["n"])

        # XXX I do not think this is useful -- is common is empty, we will never go into the FOR loop.
        if not common:
            continue

        # For each tab-separated entity mention pair ...
        for group in common:
            src, tgt = group.split("\t")

            # Get where the mentions appear in the sentence ...
            src_span = jdata["matches"][src]
            tgt_span = jdata["matches"][tgt]

            # Then retrieve the sentence itself and "sanitise" it by removing $ and ^ symbols ...
            sent = jdata["sent"]
            sent = sent.replace("$", "")
            sent = sent.replace("^", "")

            # src entity mentioned before tgt entity
            if src_span[1] < tgt_span[0]:
                # Add markers for first and second entity
                sent = sent[:src_span[0]] + "$" + src + "$" + sent[src_span[1]:tgt_span[0]] + "^" + tgt + "^" + sent[tgt_span[1]:]
                rel_dir = 1

            # tgt entity mentioned before src entity
            elif src_span[0] > tgt_span[1]:
                # Add markers for first and second entity
                sent = sent[:tgt_span[0]] + "^" + tgt + "^" + sent[tgt_span[1]:src_span[0]] + "$" + src + "$" + sent[src_span[1]:]
                rel_dir = -1

            # Should not happen, but to be on safe side
            else:
                # XXX I would add an assert False statement here
                continue

            if group not in group_to_data:
                group_to_data[group] = collections.defaultdict(list)

            # Then also keep this group_to_data dict, where group_to_data[group][rel_dir] is a list of all sentences
            # linking e1 in group with e2 (rel_dir indicates the direction of the relationship)
            group_to_data[group][rel_dir].append(sent)

    # Adjust bag sizes
    new_group_to_data: Dict[str, Dict[str, Iterable[str]]] = dict()

    # Then, for each group in group_to_data ...
    for group in list(group_to_data.keys()):
        src, tgt = group.split("\t")

        # Create a bag (list) with all sentences linking the entities in that group.
        bag = list()
        for rel_dir in group_to_data[group]:
            bag.extend(group_to_data[group][rel_dir])

        # If it exceeds some given size, we sub-sample the sentences ...
        if len(bag) > max_bag_size:
            bag = random.sample(bag, max_bag_size)
        else:
            # Otherwise, we upsample the sentences in the bag until we get to the given maximum size.
            idxs = list(np.random.choice(list(range(len(bag))), max_bag_size - len(bag)))
            bag = bag + [bag[i] for i in idxs]

        # Then the dictionary new_group_to_data will contain a mapping from groups (tab-separated entity mention pairs)
        # to "relations", containing a set with the relations linking such entities, and the newly formed bag of sents.
        new_group_to_data["\t".join([src, tgt, "0"])] = {
            "relations": group_to_relation_texts[group],
            "bag": bag
        }

    # group_to_data is now set to new_group_to_data.
    group_to_data: Dict[str, Dict[str, Iterable[str]]] = new_group_to_data

    filtered_triples: Set[Tuple[str, str, str]] = set()

    # Then, we use the relations contained in group_to_data to generate a new set of triples -- filtered_triples.
    # XXX - How is this different from the inout triples? Ask Saad.
    # One difference is that in group_to_data keys there is a "0" appended to the target. But what's the point?
    for group in group_to_data:
        src, tgt, _ = group.split("\t")
        for relation in group_to_data[group]["relations"]:
            filtered_triples.add((src, relation, tgt))

    return filtered_triples, group_to_data


def remove_overlapping_sents(train_lines: List[Dict[str, Any]],
                             dev_lines: List[Dict[str, Any]],
                             test_lines: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Iterable[str]]],
                                                                        Set[Tuple[str, str, str]]]:
    dev_test_sentences = set()

    # For each line in dev_lines:
    for line in dev_lines:
        # Look in the sentences, remove the $ and ^ symbols, and add them to dev_test_sentences
        dev_test_sentences.update({s.replace("$", "").replace("^", "") for s in line["sentences"]})

    # For each line in test_lines:
    for line in test_lines:
        # Look in the sentences, remove the $ and ^ symbols, and add them to dev_test_sentences
        dev_test_sentences.update({s.replace("$", "").replace("^", "") for s in line["sentences"]})

    # This will contain the new lines
    new_train_lines = list()

    # For each train line ..
    for line in train_lines:
        # This will contain the sentences in line that do not appear in the dev or test sets
        new_sents = list()
        # For each of its sentences ..
        for sent in line["sentences"]:
            # Remove $ and ^ symbols ..
            temp_sent = sent.replace("$", "").replace("^", "")
            # And add it to new_sents if it is not among the dev and test sentences
            if temp_sent not in dev_test_sentences:
                new_sents.append(sent)

        if not new_sents:
            continue

        # Set the number of sentences in each line to a specific value, by downsampling or upsampling
        bag = new_sents
        if len(bag) > config.bag_size:
            bag = random.sample(bag, config.bag_size)
        else:
            idxs = list(np.random.choice(list(range(len(bag))), config.bag_size - len(bag)))
            bag = bag + [bag[i] for i in idxs]

        # Replace the current bag of sentences, and add the line to new_train_lines
        line["sentences"] = bag
        new_train_lines.append(line)

    new_triples = set()

    # Create a new set of triples from new_train_lines
    for line in new_train_lines:
        src, tgt = line["group"]
        relation = line["relation"]
        new_triples.add((src, relation, tgt))

    return new_train_lines, new_triples


def create_data_split(triples: Iterable[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str, str]],
                                                                        List[Tuple[str, str, str]],
                                                                        List[Tuple[str, str, str]]]:
    triples = list(triples)
    inds = list(range(len(triples)))
    y = [relation for _, relation, _ in triples]
    # train_dev test split
    train_dev_inds, test_inds = train_test_split(inds, stratify=y, test_size=0.2, random_state=config.SEED)
    y = [y[i] for i in train_dev_inds]
    train_inds, dev_inds = train_test_split(train_dev_inds, stratify=y, test_size=0.1, random_state=config.SEED)

    train_triples = [triples[i] for i in train_inds]
    dev_triples = [triples[i] for i in dev_inds]
    test_triples = [triples[i] for i in test_inds]

    logger.info(" *** Train triples : {} *** ".format(len(train_triples)))
    logger.info(" *** Dev triples : {} *** ".format(len(dev_triples)))
    logger.info(" *** Test triples : {} *** ".format(len(test_triples)))

    return train_triples, dev_triples, test_triples


def split_lines(triples: List[Tuple[str, str, str]],
                group_to_data: Dict[str, Dict[str, Iterable[str]]]) -> List[Dict[str, Any]]:
    # Create a set of tab-separated entity mentions
    groups: Set[str] = set()
    for ei, _, ek in triples:
        groups.add("{}\t{}".format(ei, ek))

    lines: List[Dict[str, Any]] = list()

    # For each tab-separated entity mention pair ...
    for group in groups:
        src, tgt = group.split("\t")

        # G is "e1\te2\t0"
        G = ["\t".join([src, tgt, "0"]), ]
        for g in G:
            if g not in group_to_data:
                continue

            # group_to_data maps each group to two iterables: "relations" (relation names) and "bag" (sentences)
            data = group_to_data[g]
            _, _, rel_dir = g.split("\t")
            rel_dir = int(rel_dir)
            for relation in data["relations"]:
                # Each line contains:
                # - "group": (e1, e2)
                # - "relation": relation linking e1 with e2
                # - "sentences": the bag of sentences
                # - "e1": None
                # - "e2": None
                # - "reldir": direction of the relation -- in this case it's 0
                lines.append({
                    "group": (src, tgt),
                    "relation": relation,
                    "sentences": data["bag"],
                    "e1": data.get("e1", None),
                    "e2": data.get("e2", None),
                    "reldir": rel_dir
                })
    return lines


def report_data_stats(lines: List[Dict[str, Any]],
                      triples: List[Tuple[str, str, str]]):
    stats = dict(
        num_of_groups=len(lines),
        num_of_sents=sum(len(line["sentences"]) for line in lines),
        num_of_triples=len(triples)
    )
    for k, v in stats.items():
        logger.info(" *** {} : {} *** ".format(k, v))


def write_final_jsonl_file(lines: List[Dict[str, Any]],
                           output_fname: str):
    with open(output_fname, "w") as wf:
        for idx, line in enumerate(lines):
            wf.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    cui_to_entity_texts = relation_text_to_groups = None

    # 0. Load UMLS vocab object
    if not os.path.isfile(config.reltext_all_combos):
        logger.info(f"Loading UMLS vocab object `{config.umls_cui_to_txts}` ...")

        with open(config.umls_cui_to_txts, "rb") as ctt, open(config.umls_reltxt_to_groups, "rb") as rttg:
            # Keys: 'C5200875', 'C5200876', 'C5200877', ...
            # Values of cui_to_entity_texts['C5200875']: {'lnc-ru-ru_266', 'loinc, russian, russia edition, 266'}
            cui_to_entity_texts = pickle.load(ctt)

            # Keys: 'has_component', 'has_measured_component', 'measures', ...
            # Values of relation_text_to_groups['has_component']: {('C0066621', 'C1953785'), ('C0883230', 'C0942242'), ...}
            relation_text_to_groups = pickle.load(rttg)

    # 1. Collect all possible group texts from their CUIs
    # Returns a set of all tab-separated pairs of surface forms of related CUIs
    groups_texts: Set[str] = get_groups_texts_from_umls_vocab(relation_text_to_groups, cui_to_entity_texts,
                                                              config.reltext_all_combos, load_existing=True)

    # lower extremity	monoplegia of nondominant lower limb as a late effect of cerebrovascular accident (disorder)
    # paraffinum liquidum	mineral oil / phenolphthalein lozenge product
    # digestive tract structure (body structure)	crohn diseases
    # cerebrospinal fluid, nos	cv b5 ab titr csf
    # product containing acyclovir 200 mg/1 each oral capsule	eryth

    # 2. Search for text alignment of groups (this can take up to 80~90 mins)
    # Generates two sets of tab-separated pairs of CUI surface forms, one with related and one with unrelated pairs
    pos_groups, neg_groups = align_groups_to_sentences(groups_texts, # Set of tab-separated sentence pairs
                                                       config.medline_linked_sents_file,  # Input file: umls.linked_sentences.jsonl
                                                       config.groups_linked_sents_file)  # Output file: umls.linked_sentences_to_groups.jsonl

    del groups_texts
    import gc
    gc.collect()

    if relation_text_to_groups is None:
        with open(config.umls_cui_to_txts, "rb") as ctt, open(config.umls_reltxt_to_groups, "rb") as rttg:
            cui_to_entity_texts = pickle.load(ctt)
            relation_text_to_groups = pickle.load(rttg)

    # 3. From collected groups and pruning relations criteria, get final triples
    # Obtains a list of triples (e1, R, e2), where:
    #   - If R is not 'NA', e1 and e2 are mentions of entities related by the relationship R coming from pos_groups
    #   - If R is 'NA', e1 and e2 are mentions of entities coming from neg_groups
    triples = pruned_triples(cui_to_entity_texts, relation_text_to_groups, pos_groups, neg_groups,
                             config.min_rel_group, config.max_rel_group)

    # 4. Collect evidences and filter triples based on sizes of collected bags
    triples, group_to_data = filter_triples_with_evidence(triples, config.bag_size)
    # Here 'triples' is a set of (entity mention, relation name, entity mention) triples, while group_to_data links
    # each group to a set of relations linking the two entity mentions in that group, and to a list of sentences forming
    # the bag of sentences -- all bags have the same size.

    logger.info(f" *** No. of triples (after filtering) *** : {len(triples)}")

    E: Set[str] = set()
    R: Set[str] = set()

    # Write all triples to triples_file_all, which by default is data/processed/triples_all.tsv ...
    with open(config.triples_file_all, "w") as wf:
        for ei, rj, ek in triples:
            E.update([ei, ek])
            R.add(rj)
            line = "{}\t{}\t{}".format(ei, rj, ek)
            wf.write(line + '\n')

    # Write all entities to entities_file, which by default is data/processed/entities.txt ...
    with open(config.entities_file, "w") as wf:
        for e in E:
            wf.write("{}\n".format(e))

    # Write all relation types to relations_file, which by default is data/processed/relations.txt ...
    with open(config.relations_file, "w") as wf:
        for r in R:
            wf.write("{}\n".format(r))

    logger.info(" *** No. of entities *** : {}".format(len(E)))
    logger.info(" *** No. of relations *** : {}".format(len(R)))

    # 5. Split into train, dev and test at triple level to keep zero triples overlap
    # Returns three non-overlapping lists of triples
    train_triples, dev_triples, test_triples = create_data_split(triples)

    # Translate the triples in lines.
    #   Each line contains:
    #   - "group": (e1, e2)
    #   - "relation": relation linking e1 with e2
    #   - "sentences": the bag of sentences
    #   - "e1": None
    #   - "e2": None
    #   - "reldir": direction of the relation -- in this case it's 0
    train_lines = split_lines(train_triples, group_to_data)
    dev_lines = split_lines(dev_triples, group_to_data)
    test_lines = split_lines(test_triples, group_to_data)

    # Remove any overlapping test and dev sentences from training
    logger.info("Train stats before removing overlapping sentences ...")

    # Report stats.
    #   These contain:
    #   - num_of_groups: len(lines),
    #   - num_of_sents: sum(len(line["sentences"]) for line in lines),
    #   - num_of_triples: len(triples)
    report_data_stats(train_lines, train_triples)

    # Remove sentences appearing in dev or test from the training set
    train_lines, train_triples = remove_overlapping_sents(train_lines, dev_lines, test_lines)

    # Write triples file and final combined file
    logger.info("Train stats after removing dev + test overlapping sentences ...")

    triples_splits = [
        (config.complete_train, config.triples_file_train, train_triples, train_lines, "TRAIN"),
        (config.complete_dev, config.triples_file_dev, dev_triples, dev_lines, "DEV"),
        (config.complete_test, config.triples_file_test, test_triples, test_lines, "TEST")
    ]

    for complete_file, trip_file, trips, lines, split_name in triples_splits:
        print(split_name)
        report_data_stats(lines, trips)

        with open(trip_file, "w") as wf:
            for ei, rj, ek in trips:
                wf.write("{}\t{}\t{}\n".format(ei, rj, ek))

        # 6. Write train, dev, test files with sentence, group and relation
        logger.info("Creating training file at `{}` ...".format(complete_file))

        # Save the lines in complete_file as jsonl
        write_final_jsonl_file(lines, complete_file)
