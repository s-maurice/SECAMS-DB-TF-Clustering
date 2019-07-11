import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

import get_input_data

#raw_df = get_input_data.get_events_from_sql()
raw_df = get_input_data.get_events_from_csv("entries_by_user/SECAMS_user_A0001.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

le_userid = preprocessing.LabelEncoder()
userid_encoded = le_userid.fit_transform(raw_df["USERID"])

le_eventid = preprocessing.LabelEncoder()
eventid_encoded = le_eventid.fit_transform(raw_df["EVENTID"])

le_terminalsn = preprocessing.LabelEncoder()
terminalsn_encoded = le_terminalsn.fit_transform(raw_df["TERMINALSN"])


x_ticklabels = list(set(le_userid.inverse_transform(userid_encoded).tolist()))
y_ticklabels = list(set(le_eventid.inverse_transform(eventid_encoded).tolist()))
z_ticklabels = list(set(le_terminalsn.inverse_transform(terminalsn_encoded).tolist()))


plt.scatter(userid_encoded, eventid_encoded, terminalsn_encoded)

ax.set_xlabel("UserID")
ax.set_xticks(list(set(userid_encoded.tolist())))
ax.set_xticklabels(x_ticklabels)

ax.set_ylabel("EventID")
ax.set_yticks(list(set(eventid_encoded.tolist())))
ax.set_yticklabels(y_ticklabels)

ax.set_zlabel('Terminalsn')
print(list(set(terminalsn_encoded.tolist())))
ax.set_zticks(list(set(terminalsn_encoded.tolist())))
ax.set_zticklabels(z_ticklabels)

plt.show()