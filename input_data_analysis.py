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


def show_pattern(userid_list):
    # Shows a graph of the event patterns of a set of users by time and event
    # Create the plot
    fig, ax = plt.subplots(nrows=len(userid_list), sharex=True)

    fig.subplots_adjust(bottom=0.2, hspace=0.4)

    # Check if list contains single or multiple users (otherwise 'ax' will be an object instead of an array)
    if len(userid_list) == 1:
        # set title and labels for axes
        ax.set(xlabel="Date",
               ylabel="Event ID",
               title="Event logs for user " + str(userid_list[0]));

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Get data from CSV, then set x and y
        csv_filename = "CSV Files/entries_by_user/SECAMS_user_" + str(userid_list[0]) + ".csv"
        usercount_df = pd.read_csv(csv_filename)
        usercount_df['TIMESTAMPS'] = pd.to_datetime(usercount_df['TIMESTAMPS'])

        ax.scatter(x=usercount_df['TIMESTAMPS'],
                   y=usercount_df['EVENTID'])

    else:
    # Graph each user
        for userid, i in zip(userid_list, range(len(userid_list))):
            # set title and labels for axes
            ax[i].set(xlabel="Date",
                      ylabel="Event ID",
                      title="Event logs for user " + str(userid_list[i]))

            # Rotate tick labels
            plt.setp(ax[i].get_xticklabels(), rotation=45)

            # Get data from CSV, then set x and y
            csv_filename = "entries_by_user/SECAMS_user_" + str(userid) + ".csv"
            usercount_df = pd.read_csv(csv_filename)
            usercount_df['TIMESTAMPS'] = pd.to_datetime(usercount_df['TIMESTAMPS'])

            ax[i].scatter(x=usercount_df['TIMESTAMPS'],
                          y=usercount_df['EVENTID'])


show_pattern([20018])
plt.show()