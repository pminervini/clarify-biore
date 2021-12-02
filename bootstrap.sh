#!/usr/bin/env bash

mkdir -p data/processed/features/

wget -c complete_dev.txt -P data/processed/
wget -c complete_test.txt -P data/processed/
wget -c complete_train.txt -P data/processed/
wget -c complete_types_dev.txt -P data/processed/
wget -c complete_types_test.txt -P data/processed/
wget -c complete_types_train.txt -P data/processed/
wget -c entities.txt -P data/processed/
wget -c entities_types.txt -P data/processed/
wget -c features/ -P data/processed/
wget -c relations.txt -P data/processed/
wget -c relations_types.txt -P data/processed/
wget -c triples_all.tsv -P data/processed/
wget -c triples_dev.tsv -P data/processed/
wget -c triples_test.tsv -P data/processed/
wget -c triples_train.tsv -P data/processed/
wget -c triples_types_dev.tsv -P data/processed/
wget -c triples_types_test.tsv -P data/processed/
wget -c triples_types_train.tsv -P data/processed/

wget -c dev.pt -P data/processed/features/
wget -c test.pt -P data/processed/features/
wget -c train.pt -P data/processed/features/
wget -c types_dev.pt -P data/processed/features/
wget -c types_test.pt -P data/processed/features/
wget -c types_train.pt -P data/processed/features/
