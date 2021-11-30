## CLARIFY Biomedical Relation Extraction

Pipeline:

- `tools/1_process_umls.py`
  - Input: `MRREL.RRF`, `MRCONSO.RRF`, `MRSTY.RRF`
  - Output: `umls.txt_to_cui.pkl`, `umls.cui_to_txts.pkl`, `umls.reltxt_to_groups.pkl`, `umls.cui_to_types.pkl`, `umls.text_to_type.pkl`
  - Description: extracts useful information from the UMLS 2019AB files. Here, `txt` is a surface form for an entity; `reltxt` is a relation type; `group` is a pair of CUIs.

- `tools/2_split_sentences.py`
  - Input: `medline_abs.txt`
  - Output: `medline_spacy_sents.txt` (which becomes `medline_unique_sents.txt`)
  - Description: extracts sentences out of `medline_abs.txt`, which contains one MEDLINE abstract on each line.

- `tools/3_link_entities.py`
  - Input: `medline_unique_sents.txt`, `umls.txt_to_cui.pkl`
  - Output: `umls.linked_sentences.jsonl`
  - Description: identifies UMLS entities in sentences from MEDLINE abstracts, tags them, and saves the tags into `umls.linked_sentences.jsonl`.

```json
{
  "sent": "The specificity of the carcinoembryonic antigen test for identifying future colorectal cancer patients was 0.99 with a sensitivity of 0.12.",
  "matches": {
    "carcinoembryonic antigen test": [23, 52], 
    "future": [69, 75], 
    "colorectal cancer": [76, 93], 
    "patients": [94, 102], 
    "0.99": [107, 111], 
    "sensitivity": [119, 130], 
    "0.12": [134, 138]
  }
}
```

- `tools/4_generate_splits.py`
  - Input: `umls.linked_sentences.jsonl`, `umls.cui_to_txts.pkl`, `umls.reltxt_to_groups.pkl`
  - Middle: `umls.reltxt_all_combos.pkl`, `umls.linked_sentences_to_groups.jsonl`
  - Output: `triples_all.tsv`, `complete_train/dev/test.txt`, `triples_train/dev/test.tsv`, `entities.txt`, `relations.txt`
  - Description: generates the train/validation/test splits. While doing so, it generates `umls.reltxt_all_combos.pkl`, which is a large set of `e1\te2` strings, where `e1` and `e2` are the surface forms of two entities that are linked together. It also generates `umls.linked_sentences_to_groups.jsonl`.

Each element in `umls.linked_sentences_to_groups.jsonl` looks like this, where `groups` contains positive and negative examples of related entity pairs, where negative examples are not related in the Knowledge Base.

```json
{
  "sent": "To investigate the medium term outcome of cardiac surgery, we evaluated patients over 75 years of age who were operated on within a 1.5-year period.",
  "matches": {
    "medium": [
      19,
      25
    ],
    "term": [
      26,
      30
    ],
    "outcome": [
      31,
      38
    ],
    "cardiac surgery": [
      42,
      57
    ],
    "evaluated": [
      62,
      71
    ],
    "patients": [
      72,
      80
    ],
    "years": [
      89,
      94
    ],
    "age": [
      98,
      101
    ],
    "within": [
      123,
      129
    ],
    "1.5": [
      132,
      135
    ],
    "year": [
      136,
      140
    ],
    "period": [
      141,
      147
    ]
  },
  "groups": {
    "p": [
      "age\tpatients",
      "patients\tage"
    ],
    "n": [
      "patients\t1.5",
      "patients\tyears"
    ]
  }
}
```

Then, `triples_all.tsv` looks like the following:

```tsv
adiponectin	genetic_biomarker_related_to	acrp30
dose	has_system	methadone dose
celiac angiography	substance_used_by	contrast
physical therapy education	focus_of	physical therapy
autosomal dominant	has_inheritance_type	obese
```

And finally, `complete_train/dev/test.txt` looks like the following:

```json
{
  "group": ["screw loosening", "plate fracture"], 
  "relation": "temporally_followed_by", 
  "sentences": [
    "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24\u00a0h, six weeks and three months postoperatively.",
    "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
    "From a review of the literature, it is evident that the technique used most frequently for fixation is the positioning of a single plate despite complications concerning ^plate fracture^ or $screw loosening$ have been reported by various authors.",
    "In 52% of patients, fracture healing was uneventful; however, in 48% of patients, complications were encountered, including osteomyelitis, nonunion, ^plate fracture^, $screw loosening$, and dehiscences with subsequent infections.",
    "The use of second-generation locking reconstruction plates for the treatment of mandibular continuity defects has a 36% complication rate, which includes ^plate fracture^, $screw loosening$, plate exposure, wound infection and malocclusion.",
    "In 52% of patients, fracture healing was uneventful; however, in 48% of patients, complications were encountered, including osteomyelitis, nonunion, ^plate fracture^, $screw loosening$, and dehiscences with subsequent infections.",
    "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
    "The use of second-generation locking reconstruction plates for the treatment of mandibular continuity defects has a 36% complication rate, which includes ^plate fracture^, $screw loosening$, plate exposure, wound infection and malocclusion.",
    "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24\u00a0h, six weeks and three months postoperatively.",
    "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    "The use of second-generation locking reconstruction plates for the treatment of mandibular continuity defects has a 36% complication rate, which includes ^plate fracture^, $screw loosening$, plate exposure, wound infection and malocclusion.",
    "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months."],
  "e1": null,
  "e2": null,
  "reldir": 0
}
```

- `tools/5_generate_name_to_type_splits.py`
  - Input: `complete_train/dev/test.txt`
  - Output: `complete_types_train/dev/test.txt`, `triples_train/dev/test.tsv`, `triples_types_train/dev/test.tsv`
  - Description: generates the splits but including the entity types rather than just the entities.

More specifically, each line in `complete_types_train/dev/test.txt` looks like this:

```json
{
  "group": [
    "pathologic function",
    "therapeutic or preventive procedure"
  ],
  "relation": "temporally_followed_by",
  "ent_names": [
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "screw loosening",
      "plate fracture"
    ],
    [
      "surgical site infections",
      "surgical procedure"
    ],
    [
      "surgical site infections",
      "surgical procedure"
    ],
    [
      "surgical site infections",
      "surgical procedure"
    ],
    [
      "surgical site infections",
      "surgical procedure"
    ],
    [
      "surgical site infections",
      "surgical procedure"
    ],
    [
      "surgical site infection",
      "procedures"
    ],
    [
      "surgical site infection",
      "procedures"
    ],
    [
      "surgical site infection",
      "procedures"
    ],
    [
      "surgical site infection",
      "procedures"
    ]
  ],
  "sentences": [
    "Radiographically, the approximation of fracture fragments, ^plate fracture^ and $screw loosening$ on orthopantomograph and Reverse Towne's view were evaluated at intervals of 24 h, six weeks and three months postoperatively.",
    "It is important to establish the time-related risk of complications such as ^plate fracture^ or $screw loosening$.",
    "The average time frame until a hardware failure (^plate fracture^, $screw loosening$) occurs is 14 months.",
    "We found 11.3% complications due to ^plate fracture^, plate torsion, or $screw loosening$.",
    "From a review of the literature, it is evident that the technique used most frequently for fixation is the positioning of a single plate despite complications concerning ^plate fracture^ or $screw loosening$ have been reported by various authors.",
    "In 52% of patients, fracture healing was uneventful; however, in 48% of patients, complications were encountered, including osteomyelitis, nonunion, ^plate fracture^, $screw loosening$, and dehiscences with subsequent infections.",
    "The use of second-generation locking reconstruction plates for the treatment of mandibular continuity defects has a 36% complication rate, which includes ^plate fracture^, $screw loosening$, plate exposure, wound infection and malocclusion.",
    "This secondary analysis evaluated the association of operating room scrub staff expertise, based on frequency of working on a specific ^surgical procedure^, with the development of $surgical site infections$.",
    "The risk factors for $surgical site infections$ in surgery should be measured and monitored from admission to 30 days after the ^surgical procedure^, because 30% of Surgical Site Infection is detected when the patient was discharged.",
    "Risk factors for the occurrence of $surgical site infections$ include variables related to the ^surgical procedure^ as well as host factors.",
    "On occasion, the placement of orthopedic prosthetic components or stabilization hardware leads to $surgical site infections$, in some cases presenting at a point in time distant from the ^surgical procedure^.",
    "The aim of our study was to evaluate the effect of stoma creation on deep and superficial $surgical site infections$ after an index colorectal ^surgical procedure^.",
    "Preliminary results obtained for two indicator ^procedures^ show no significant differences in $surgical site infection$ rates between outpatient surgery institutions and the hospital setting (OP-KISS).",
    "Eight of the 16 Dutch cardiothoracic centers participated and collected data on 4066 ^procedures^ and 183 surgical site infections, revealing a $surgical site infection$ rate of 2.4% for sternal wounds and 3.2% for harvest sites.",
    "VP shunting and tracheotomy ^procedures^ could be performed simultaneously without increasing the risk of $surgical site infection$.",
    "We retrospectively reviewed the cost of return ^procedures^ for treatment of $surgical site infection$ (SSI)."
  ]
}
```

While each line in  `triples_types_train/dev/test.tsv` looks like this:

```tsv
body part, organ, or organ component	contains	body part, organ, or organ component
biomedical or dental material	state_of_matter_of	chemical viewed structurally
laboratory procedure	has_modification	vitamin
body substance	is_location_of_anatomic_structure	neoplastic process
molecular function	is_mechanism_of_action_of_chemical_or_drug	cell function
therapeutic or preventive procedure	evaluation_of	intellectual product
behavior	answer_to	qualitative concept
gene or genome	is_abnormality_of_gene	cell or molecular dysfunction
finding	answer_to	functional concept
body space or junction	superior_to	body space or junction
```

- `tools/6_generate_features.py`
  - Input: `entities.txt`, `relations.txt`, `complete_train/dev/test.txt`
  - Output: `train/dev/test.pt`

- `tools/7_generate_abstracted_features.py`
  - Input: `entities.txt`, `relations.txt`, `complete_types_train/dev/test.txt`
  - Output: `types_train/dev/test.pt`
