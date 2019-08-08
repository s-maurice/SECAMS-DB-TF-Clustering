import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("C://Users//Maurice//PycharmProjects//SECAMS DB TF Clustering//absence_detect//reason_df.csv")
df["USERID"] = [str(i) for i in df["USERID"]]
df['Day'] = pd.to_datetime(df['Day'])


plt.figure("Generated Reasons Distribution", figsize=(10,7))
ax0 = plt.subplot(111)
reason_dict = df["Reason"].value_counts()
rects = plt.bar(reason_dict.index, reason_dict.values)
plt.yscale("linear")
plt.title("Generated Event Reason Distribution")
plt.setp(ax0.get_xticklabels(), ha="center", rotation=90)
plt.xlabel("Reason")
plt.ylabel("Frequency")
plt.ylim(0, 272000)

for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
            '%d\n%.2f%%' % (int(height), int(height)/len(df)*100),
            ha='center', va='bottom')

plt.subplots_adjust(bottom=0.24)
plt.savefig('freq_bar.png', dpi=500)

fig2 = plt.figure("UserID by Day", figsize=(10,5))
ax = plt.subplot(111)

userid_le = preprocessing.LabelEncoder()
df["USERID"] = userid_le.fit_transform(df["USERID"])
df["Day"] = [i.day + 30 * (i.month-5) for i in df["Day"]]

for reason, color in zip(df["Reason"].unique(), range(len(df["Reason"].unique()))):
    plt.scatter(df.loc[df["Reason"] == reason]["Day"], df.loc[df["Reason"] == reason]["USERID"], marker=".", label=reason)


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
plt.xlabel("Day, starting from 1st May, 2016")
plt.ylabel("UserID")
plt.title("UserID Time Events")
plt.savefig('userid_time_events.png', dpi=500)
plt.show()