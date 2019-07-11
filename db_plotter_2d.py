import pandas as pd
import matplotlib.pyplot as plt

import get_input_data

raw_df = get_input_data.get_events_from_csv("CSV Files/Curated Data/ALL_USERID_beginning_with_20_and_between_100_and_500_entries.csv")

plt.scatter(raw_df["TIMESTAMPS"], raw_df["USERID"])
plt.show()