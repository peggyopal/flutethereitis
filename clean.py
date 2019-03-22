"""
File Name: clean.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 21-03-2019

Description: A script to 'clean' the data sets to be smaller and more precise
             for our use cases

"""

import os
import sys

module_path = os.path.dirname(os.path.abspath("src/dataset_characterization"))
sys.path.insert(0, module_path + '/../')
import src.dataset_characterization.clean_embeddings as ce
import src.dataset_characterization.clean_labels as label
import src.dataset_characterization.clean_segment as cs


print("=================================")
print("Cleaning Labels")
label.clean_all_labels()
print("Done Cleaning the Labels - That was tense!")
print("\n")


print("=================================")
print("Cleaning Segments")

print("Balanced Train")
cs.clean_bal_train_segments()

print("Unbalanced Train")
cs.clean_unbal_train_segments()

print("Eval")
cs.clean_eval_segments()
print("Done Cleaning Segments - Almost There...")
print()


print("=================================")
print("Cleaning TensorRecords")
print("Balanced Train")
ce.clean_bal_train_records()

print("Unbalanced Train")
ce.clean_unbal_train_records()

print("Eval")
ce.clean_eval_records()

print("YAY! ALL DONE!.")
print("=================================")
