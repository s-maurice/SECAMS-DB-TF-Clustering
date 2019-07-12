import pandas as pd
import matplotlib.pyplot as plt

import get_input_data


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


raw_df = get_input_data.get_events_from_csv("CSV Files/Curated Data/ALL_USERID_beginning_with_20_and_between_100_and_500_entries.csv")
print(raw_df.sort_values(by=["TIMESTAMPS"]))
#raw_df = raw_df.loc[raw_df['USERID'] == 20034].copy()
#raw_df2 = raw_df.loc[raw_df['USERID'] == 20085].copy()
#raw_df = raw_df.groupby('TERMINALSN').filter(lambda x: len(x) > 20)
#raw_df.to_csv("CSV Files/Curated Data/ALL_USERID_beginning_with_20_and_between_100_and_500_entries.csv")
raw_df2 = raw_df

plt.subplot(211)
raw_df2["DECHOUR"] = get_decimal_hour(raw_df2["TIMESTAMPS"])
plt.scatter(raw_df2["DECHOUR"], raw_df2["TERMINALSN"])

plt.subplot(212)
raw_df["DECHOUR"] = get_decimal_hour(raw_df["TIMESTAMPS"])
plt.scatter(raw_df["DECHOUR"], raw_df["TERMINALSN"])
plt.show()