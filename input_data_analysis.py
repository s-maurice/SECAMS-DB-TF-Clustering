import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_usercounts():
    # Shows a cumulative graph of users with a certain number of entries/logs in the database
    usercount_df = pd.read_csv("info_usercount.csv")
    print(usercount_df.describe())

    plt.hist(
        usercount_df['usercount'],
        bins=list(range(0, 6000, 10)),
        cumulative=False)
    plt.yscale('log')
    plt.axis([0, 6000, 0, 10000])  # Lock axis
    plt.show()


def show_pattern(userid):
    # Shows a graph of the event patterns of a particular user by time and event
    csv_filename = "entries_by_user/SECAMS_user_" + str(userid) + ".csv"
    usercount_df = pd.read_csv(csv_filename)
    usercount_df['TIMESTAMPS'] = pd.to_datetime(usercount_df['TIMESTAMPS'])

    print(usercount_df.columns)

    usercount_df.plot(x='TIMESTAMPS', y='EVENTID')



show_pattern(20018)
plt.show()