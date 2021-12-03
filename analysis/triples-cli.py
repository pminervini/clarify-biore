#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm

from typing import Dict, Set, Tuple, Generator


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


def main(argv):
    mrr_path = 'data/UMLS/raw/MRREL.RRF'

    source_to_triple_set: Dict[str, Set[Tuple[str, str, str]]] = dict()

    for s, p, o, source in tqdm(extract_triples(mrr_path)):
        if source not in source_to_triple_set:
            source_to_triple_set[source] = set()
        source_to_triple_set[source] |= {(s, p, o)}

    for source, triples in tqdm(source_to_triple_set.items()):
        lines = sorted([f'{s}\t{p}\t{o}\n' for s, p, o in triples])
        with open(f'analysis/data/triples/{source}.tsv', 'w') as f:
            f.writelines(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
