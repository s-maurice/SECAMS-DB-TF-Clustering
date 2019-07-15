import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from numpy import datetime64
from more_itertools import unique_everseen

import get_input_data

# Get and prepare DF
raw_df = get_input_data.get_events_from_csv("CSV Files/Curated Data/ALL_USERID_beginning_with_20_and_between_100_and_500_entries.csv")
# raw_df = get_input_data.get_events_from_csv("CSV Files/Curated Data/userid_20xxx_terminal_400up_user_100to500_hour_15down.csv")


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


def remove_duplicates(array):
    return list(unique_everseen(array))


if "TIMESTAMPS" in raw_df.columns:
    raw_df["TIMESTAMPS"] = raw_df["TIMESTAMPS"].apply(datetime64)
    raw_df["DECHOUR"] = get_decimal_hour(raw_df["TIMESTAMPS"])

# Figures + subplots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x: user // y: timestamp // z: terminalsn // color: eventID
# 3d plots can't deal with categorical (x and z); encode using label encoders
le_userid = preprocessing.LabelEncoder()
le_terminalsn = preprocessing.LabelEncoder()

# x
userid_encoded = le_userid.fit_transform(raw_df["USERID"])
x_ticklabels = remove_duplicates(le_userid.inverse_transform(userid_encoded).tolist())

# y
dechour = raw_df["DECHOUR"].values

# z
terminalsn_encoded = le_terminalsn.fit_transform(raw_df["TERMINALSN"])
z_ticklabels = remove_duplicates(le_terminalsn.inverse_transform(terminalsn_encoded).tolist())

# colormap
le_eventid = preprocessing.LabelEncoder()
eventid_encoded = le_eventid.fit_transform(raw_df["EVENTID"])

# print(le_eventid.inverse_transform(eventid_encoded).tolist())
print(remove_duplicates(le_eventid.inverse_transform(eventid_encoded).tolist()))

p = ax.scatter(userid_encoded, dechour, terminalsn_encoded, c=eventid_encoded, cmap="Set1")


ax.set_xlabel("UserID")
ax.set_xticks(remove_duplicates(userid_encoded.tolist()))
ax.set_xticklabels(x_ticklabels)

ax.set_ylabel("Time of day")

ax.set_zlabel('TerminalSN')
ax.set_zticks(remove_duplicates(terminalsn_encoded.tolist()))
ax.set_zticklabels(z_ticklabels)


# Give the colorbar - ticks doesn't work on Mac
fig.colorbar(p)
# fig.colorbar(p, ticks=list(set(le_eventid.inverse_transform(eventid_encoded).tolist())))
plt.show()


# deprecated: EVENTID as y axis
# y_ticklabels = list(set(le_eventid.inverse_transform(eventid_encoded).tolist()))
# ax.set_yticks(list(set(eventid_encoded.tolist())))
# ax.set_yticklabels(y_ticklabels)
