#!/bin/sh

python clean_labels.py

python clean_segment.py ../../data/balanced_train_segments.csv
python clean_segment.py ../../data/eval_segments.csv
python clean_segment.py ../../data/unbalanced_train_segments.csv

python clean_embeddings.py ../../data/audioset_v1_embeddings/bal_train ../../data/balanced_train_segments_clean.csv
python clean_embeddings.py ../../data/audioset_v1_embeddings/unbal_train ../../data/unbalanced_train_segments_clean.csv
python clean_embeddings.py ../../data/audioset_v1_embeddings/eval ../../data/eval_segments_clean.csv
