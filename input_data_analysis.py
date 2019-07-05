import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

usercount_df = pd.read_csv("info_usercount.csv")
print(usercount_df.describe())

plt.hist(
    usercount_df['usercount'],
    bins=list(range(0, 6000, 10)),
    cumulative=False)
plt.yscale('log')
plt.axis([0, 6000, 0, 10000])  # Lock axis
plt.show()

# Create a dataframe to hold the number of people with a certain amount of event logs
# key: number of events
# value: number of people
usercount_plot_df = pd.DataFrame(columns=['key','value'])

# usercount_df["usercount"].plot()
# plt.show()

# for userid, count in usercount_df.iterrows():
#
