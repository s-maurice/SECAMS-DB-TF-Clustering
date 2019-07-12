import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import datetime64


def show_usercount():
    # Shows a cumulative graph of users with a certain number of entries/logs in the database
    # Uses CSV Files/info_usercount.csv specifically.
    usercount_df = pd.read_csv("CSV Files/info_usercount.csv")
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
    # Uses 'CSV Files/entries_by_user' specifically.
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


def plotpoints(filepath, x, y):
    # A method that simply plots one variable against another, given the appropriate CSV file and column names.
    # (Takes in 3 strings.)

    # Read the CSV filepath for DataFrame
    df = pd.read_csv(filepath)

    def get_decimal_hour(events):
        decimal_hour = (events.dt.hour + events.dt.minute / 60)
        return decimal_hour

    # Fix the DF
    if "TIMESTAMPS" in df.columns:
        df["TIMESTAMPS"] = df["TIMESTAMPS"].apply(datetime64)
        df["DECHOUR"] = get_decimal_hour(df["TIMESTAMPS"])
    if "USERID" in df.columns:
        df["USERID"]=df["USERID"].apply(lambda x: str(x))

    fig, ax = plt.subplots()
    ax.set(xlabel=x,
           ylabel=y)

    ax.plot(df[x], df[y], marker='o', linestyle='')


def main():
    plotpoints(filepath="CSV Files/Curated Data/userid_20xxx_terminal_400up_user_100up_hour_15down.csv",
           x="DECHOUR",
           y="USERID")
    # show_pattern([20018])


main()
plt.show()
