#!/usr/bin/env bash

mkdir -p data/processed/features/

wget -c http://data.neuralnoise.com/amil/data/processed/complete_dev.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/complete_test.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/complete_train.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/complete_types_dev.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/complete_types_test.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/complete_types_train.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/entities.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/entities_types.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/features/ -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/relations.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/relations_types.txt -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_all.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_dev.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_test.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_train.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_types_dev.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_types_test.tsv -P data/processed/
wget -c http://data.neuralnoise.com/amil/data/processed/triples_types_train.tsv -P data/processed/

wget -c http://data.neuralnoise.com/amil/data/processed/features/dev.pt -P data/processed/features/
wget -c http://data.neuralnoise.com/amil/data/processed/features/test.pt -P data/processed/features/
wget -c http://data.neuralnoise.com/amil/data/processed/features/train.pt -P data/processed/features/
wget -c http://data.neuralnoise.com/amil/data/processed/features/types_dev.pt -P data/processed/features/
wget -c http://data.neuralnoise.com/amil/data/processed/features/types_test.pt -P data/processed/features/
wget -c http://data.neuralnoise.com/amil/data/processed/features/types_train.pt -P data/processed/features/
