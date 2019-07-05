import pandas as pd
import matplotlib.pyplot as plt

usercount_df = pd.read_csv("info_usercount.csv")
count_unique = usercount_df['usercount'].unique()

usercount_df["usercount"].plot()
plt.show()
#
# usercount_dict = dict()
#
# for userid, count in usercount_df.iterrows():
#     if count in usercount_dict:
#         # increase the value by 1
#
#     else:
#         # make the key, and assign it a value of one