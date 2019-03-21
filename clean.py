"""
File Name: clean.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 21-03-2019

Description: A script to 'clean' the data sets to be smaller and more precise
             for our use cases

"""

module_path = os.path.dirname(os.path.abspath("src/helpers.py"))
sys.path.insert(0, module_path + '/../')
import src.helpers as help


print("=================================")
print("Cleaning Labels")
help.clean_all_labels()
print("Done Cleaning the Labels - That was tense!")
print("\n")


print("=================================")
print("Cleaning Segments")

print("Balanced Train")
help.clean_bal_train_segments()

print("Unbalanced Train")
help.clean_unbal_train_segments()

print("Eval")
help.clean_eval_segments()
python clean_segment.py ../../data/eval_segments.csv
print("Done Cleaning Segments - Almost There...")
print()


print("=================================")
print("Cleaning TensorRecords")
print("Balanced Train")
help.clean_bal_train_records()

print("Unbalanced Train")
help.clean_unbal_train_records()

print("Eval")
help.clean_eval_records()

print("YAY! ALL DONE!.")
print("=================================")
