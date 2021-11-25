#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import json
import logging
import pickle
import time

from flashtext import KeywordProcessor

import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ExactEntityLinking:

    def __init__(self, entities, case_sensitive=False):
        self.linker = KeywordProcessor(case_sensitive=case_sensitive)
        self.n_no_spans = 0
        self.n_overlapping_spans = 0
        self.n_multiple_ents = 0
        logger.info(f"Case sensitive entities: {case_sensitive}")
        logger.info(f"Building data structure with flashText for exact match entity linking (|E|={len(entities)}) ...")
        t = time.time()
        self.linker.add_keywords_from_list(list(set(entities)))
        logger.info(f"Took {(time.time() - t) // 60} mins")

    def link(self, text: str):
        spans = sorted([(start_span, end_span) for _, start_span, end_span in self.linker.extract_keywords(text, span_info=True)], key=lambda span: span[0])
        if not spans:
            self.n_no_spans += 1
            return

        # Remove overlapping matches, if any
        filtered_spans = list()
        for i in range(1, len(spans)):
            span_prev, span_next = spans[i - 1], spans[i]
            if span_prev[1] < span_next[0]:
                filtered_spans.append(spans[i])
            else:
                self.n_overlapping_spans += 1
        spans = filtered_spans[:]

        matches_texts = [text[s:e] for s, e in spans]

        # Check if any entity is present more than once, drop this sentence
        counts = collections.Counter(matches_texts)
        skip = False

        for _, count in counts.items():
            if count > 1:
                skip = True
                self.n_multiple_ents += 1
                break
        if skip:
            return

        text2span = {matches_texts[i]: spans[i] for i in range(len(spans))}

        return text2span


def link_sentences(linker: ExactEntityLinking, sents_fname: str, output_fname: str):
    sent_too_short, sent_too_long = 0, 0

    with open(sents_fname, encoding="utf-8", errors="ignore") as rf, open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
        for idx, sent in enumerate(rf):
            if idx % 1000000 == 0 and idx != 0:
                logger.info(f"Checked {idx} sentences for entity linking")
            sent = sent.strip()

            # Skip short or very long sentences
            if (len(sent) < config.min_sent_char_len_linker):
                sent_too_short += 1
                continue
            if (len(sent) > config.max_sent_char_len_linker):
                sent_too_long += 1
                continue

            # Link sentence with found ents
            text2span = linker.link(sent)
            if text2span is None:
                continue
            jdata = {"sent": sent, "matches": text2span}
            wf.write(json.dumps(jdata) + "\n")

    # Link stats:
    logger.info(f"\n\n{'+' * 20} LINKER STATS {'+' * 20}")
    if 'nltk' in sents_fname:
        logger.info("Sentences loaded from NLTK preprocessing step.")
    elif 'spacy' in sents_fname:
        logger.info("Sentences loaded from SciSPACY preprocessing step.")
    logger.info(f"Sents too short (shorter than {config.min_sent_char_len_linker}): {sent_too_short}, "
                f"Sents too long (longer than {config.max_sent_char_len_linker}): {sent_too_long}")
    logger.info(f"Number of no-span sentences: {linker.n_no_spans}")
    logger.info(f"Number of sentences with overlapping spans: {linker.n_overlapping_spans}")
    logger.info(f"Number of sentences with multiple of the same entity: {linker.n_multiple_ents}")


if __name__ == "__main__":
    with open(config.umls_txt_to_cui, "rb") as rf:
        txt_to_cui = pickle.load(rf)
    linker = ExactEntityLinking(txt_to_cui.keys())
    link_sentences(linker, config.medline_unique_sents_file, config.medline_linked_sents_file)
