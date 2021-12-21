#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from tqdm import tqdm

from typing import Dict, Set, Tuple, Generator

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def extract_triples(path: str, ro_only: bool = True) -> Generator[Tuple[str, str, str, str], None, None]:
    """Reads UMLS relation triples file MRREL.2019.RRF.

    Use ``ro_only`` to consider relations of "RO" semantic type only.
    50813206 lines in UMLS2019.

    RO = has relationship other than synonymous, narrower, or broader

    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/?report=objectonly
    """
    with open(path) as rf:
        for line in rf:
            line = line.strip()

            if not line:
                continue

            # Each line is as such: C0012792|A24166664|SCUI|RO|C0026827|A0088733|SCUI|induced_by|R176819430||MED-RT|MED-RT||N|N||
            line = line.split("|")

            # Consider relations of 'RO' type only
            if line[3] != "RO" and ro_only:
                continue

            s: str = line[0].strip()
            o: str = line[4].strip()
            p: str = line[7].strip()
            source: str = line[10].strip()

            # considering relations with textual descriptions only
            if not p:
                continue

            yield s, p, o, source
    return


def extract_types(mrsty_file: str) -> Generator[Tuple[str, str], None, None]:
    """Reads UMLS semantic types file MRSTY.2019.RRF.
    For details on each column, please check: https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """
    with open(mrsty_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            # Each line is as such:
            # 'C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347|256|
            # CUI|TUI|STN|STY|ATUI|CVF
            # Unique identifier of concept|Unique identifier of Semantic Type|Semantic Type tree number|Semantic Type.
            # The valid values are defined in the Semantic Network.|Unique identifier for attribute|Content View Flag
            line = line.split("|")

            e_id = line[0]
            e_type = line[3].strip()

            # considering entities with entity types only
            if not e_type:
                continue

            yield e_id, e_type
    return


def extract_names(mrconso_file: str,
                  en_only: bool = True,
                  lower_case: bool = True) -> Generator[Tuple[str, str], None, None]:
    """Reads UMLS concept names file MRCONSO.2019.RRF.

    Use ``en_only`` to read English concepts only.
    11743183 lines for UMLS2019.

    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly

    """
    with open(mrconso_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            # Each line is as such:  C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|
            line = line.split("|")

            # Consider en only
            if line[1] != "ENG" and en_only:
                continue

            e_id = line[0]
            e_text = line[-5].strip()

            if not e_text:
                continue

            if lower_case:
                e_text = e_text.lower()

            yield e_id, e_text
    return


def main(argv):
    mrrel_path = 'data/UMLS/raw/MRREL.RRF'
    mrsty_path = 'data/UMLS/raw/MRSTY.RRF'
    mrconso_path = 'data/UMLS/raw/MRCONSO.RRF'

    logger.info('Extracting names ..')

    cui_to_names: Dict[str, Set[str]] = dict()

    for e, name in tqdm(extract_names(mrconso_path)):
        if e not in cui_to_names:
            cui_to_names[e] = set()
        cui_to_names[e] |= {name}

    logger.info('Extracting triples ..')

    source_to_triple_set: Dict[str, Set[Tuple[str, str, str]]] = dict()

    for s, p, o, source in tqdm(extract_triples(mrrel_path)):
        if source not in source_to_triple_set:
            source_to_triple_set[source] = set()
        source_to_triple_set[source] |= {(s, p, o)}

    logger.info('Extracting types ..')

    for s, o in tqdm(extract_types(mrsty_path)):
        if 'types' not in source_to_triple_set:
            source_to_triple_set['types'] = set()
        source_to_triple_set['types'] |= {(s, 'type', o)}

    logger.info('Writing triples ..')

    for source, triples in tqdm(source_to_triple_set.items()):
        lines = sorted([f'{s}\t{p}\t{o}\n' for s, p, o in triples])
        with open(f'analysis/data/triples/{source}.tsv', 'w') as f:
            f.writelines(lines)

    logger.info('Writing readable triples ..')

    def cui_to_name(e: str) -> str:
        e_names = sorted(cui_to_names[e])
        return e_names[0]

    for source, triples in tqdm(source_to_triple_set.items()):
        lines = sorted([f'{cui_to_name(s)}\t{p}\t{cui_to_name(o)}\n' for s, p, o in triples
                        if s in cui_to_names and o in cui_to_names])
        with open(f'analysis/data/triples/{source}_readable.tsv', 'w') as f:
            f.writelines(lines)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
