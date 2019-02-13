#!/bin/sh

echo "================================="
echo "Cleaning Labels"
python clean_labels.py
echo "\nDone."
echo "\n\n"

echo "================================="
echo "Cleaning Balanced Segments"

python clean_segment.py ../../data/balanced_train_segments.csv
echo "\nDone."
echo "\n\n"


echo "================================="
echo "Cleaning Eval Segments"

python clean_segment.py ../../data/eval_segments.csv
echo "\nDone."
echo "\n\n"


echo "================================="
echo "Cleaning Unbalanced Segments "

python clean_segment.py ../../data/unbalanced_train_segments.csv
echo "\nGroovy!"
echo "\n\n"


echo "================================="
echo "Cleaning Balanced Tensors"

python clean_embeddings.py ../../data/audioset_v1_embeddings/bal_train ../../data/balanced_train_segments_clean.csv
echo "\nThat was tense!"
echo "\n\n"


echo "================================="
echo "Cleaning Unbalanced Tensors"

python clean_embeddings.py ../../data/audioset_v1_embeddings/unbal_train ../../data/unbalanced_train_segments_clean.csv
echo "\nAlmost there..."
echo "\n\n"


echo "================================="
echo "Cleaning Eval Tensors"

python clean_embeddings.py ../../data/audioset_v1_embeddings/eval ../../data/eval_segments_clean.csv
echo "\nYAY! ALL DONE!."
echo "\n\n"


