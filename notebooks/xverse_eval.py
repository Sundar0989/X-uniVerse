# %%
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos

from xverse.ensemble import VotingSelector
from xverse.feature_subset import SplitXY
from xverse.transformer import WOE, MonotonicBinning

# %%
df = pd.read_csv("./data/bank.csv", sep="|")
clf = SplitXY(["target"])  # Split the dataset into X and y
X, y = clf.fit_transform(
    df
)  # returns features (X) dataset and target(Y) as a numpy array

# %%
clf = MonotonicBinning()
clf.fit(X, y)


# %%
def prep_dataset():
    df = pd.read_csv("./data/bank.csv", sep="|")

    from xverse.feature_subset import SplitXY

    clf = SplitXY(["target"])  # Split the dataset into X and y
    X, y = clf.fit_transform(
        df
    )  # returns features (X) dataset and target(Y) as a numpy array

    return X, y


# %%
output_X = clf.transform(X)
# %%
output_bins = clf.bins
# %%
output_X
# %%
clf = MonotonicBinning(custom_binning=output_bins)  # output_bins was created earlier
out_X = clf.transform(X)
out_X.head()


# %%
clf = WOE()
clf.fit(X, y)
clf.woe_df.head()
# %%
clf.iv_df
# %%
clf.woe_bins

# %%
output_mono_bins = clf.mono_custom_binning  # future transformation
output_mono_bins
# %%
# X, y = prep_dataset()
# X, y

# # %%
# clf = VotingSelector()
# # %%
# clf.fit(X, y)
