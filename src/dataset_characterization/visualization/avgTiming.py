import matplotlib.pyplot as plt
import pandas as pd
import os

PATH_TO_CLEAN_DATA_FOLDER = "../../../data/clean_data"

# Read the cleaned version of the labels
unbal_train = pd.read_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, "unbalanced_train_segments_cleaned.csv"))

unbal_train_flute = unbal_train[unbal_train["positive_labels"] == "/m/0l14j_"]
unbal_train_didg = unbal_train[unbal_train["positive_labels"] == "/m/02bxd"]
# print(unbal_train_flute)
# print(unbal_train_didg)

length_of_time_flute = unbal_train_flute["end_seconds"]-unbal_train_flute["start_seconds"]
print("Flute -----------------------------------------")
print("Average Length of Time: ", length_of_time_flute.mean())
print("Length of Time Less Than 10 Seconds: ", length_of_time_flute[length_of_time_flute < 10.0])

length_of_time_didg = unbal_train_didg["end_seconds"]-unbal_train_didg["start_seconds"]
print("Didgeridoo -----------------------------------------")
print("Average Length of Time: ", length_of_time_didg.mean())
print("Length of Time Less Than 10 Seconds: ", length_of_time_didg[length_of_time_didg < 10.0])

# bal_train_counts.plot.bar()
#bal_train_counts.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')

#plt.grid(axis='y')
# plt.show()


# import os
#
#
# PATH_TO_CLEAN_DATA_FOLDER = "../../../data/clean_data"
#
#
# def calc_counts(csv):
#     flute = didg = both = 0
#
#     file = open(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, csv), 'r')
#     file.readline()
#
#     for line in file:
#         line = line.rstrip().split(",")
#         flute += 1 if line[4] == "/m/0l14j_" else 0
#         didg += 1 if line[4] == "/m/02bxd" else 0
#         both += 1 if line[4] == "/m/0l14j_,/m/02bxd" else 0
#     file.close()
#
#     return flute, didg, both
#
#
#
# bal_train_flute, bal_train_didg, bal_train_both = calc_counts("balanced_train_segments_cleaned.csv")
# unbal_train_flute, unbal_train_didg, unbal_train_both = calc_counts("unbalanced_train_segments_cleaned.csv")
# eval_flute, eval_didg, eval_both = calc_counts("unbalanced_train_segments_cleaned.csv")
#
# hist_values = [bal_train_flute, bal_train_didg, bal_train_both, unbal_train_flute, unbal_train_didg, unbal_train_both, eval_flute, eval_didg, eval_both]
# # print(hist_values)
# nbins = 9
#
# plt.figure()
# # plt.hist(hist_values)#, color=['cyan', 'magenta', 'black'])
# plt.hist(bal_train_flute)#, color=['cyan', 'magenta', 'black'])
# plt.ylabel("Total Number of Samples")
# # plt.yticks(range(0,130, 5))
# plt.show()
