import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv("AppliancesCsv.csv", low_memory=False)

nona = data.dropna(axis=0, how="any")

overall = data.loc[:, "overall"]
review_text = data.loc[:, "reviewText"]
summary = data.loc[:, "summary"]
vote = data.loc[:, "vote"]
verified = data.loc[:, "verified"]
# image = data.loc[:, "image"]

# review_num_adj = data.loc[:, "review_num_adj"]
# summary_num_adj = data.loc[:, "summary_num_adj"]
# review_num_len = data.loc[:, "review_num_length"]
# summary_num_len = data.loc[:, "summary_num_length"]


# features = [overall, review_text, summary, vote, verified]
counter = 0
review_text_len = []

for entry in review_text:
    review_text_len.append(len(str(entry)))

plt.scatter(overall, review_text_len)
plt.show()
# for x in features:
#     for y in features:
#         plt.scatter(x, y)
#         plt.savefig(f"plots/{counter}.png")
#         counter += 1


