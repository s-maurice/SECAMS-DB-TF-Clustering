import pyodbc
import pandas as pd
from numpy import datetime64


def get_events_from_sql():
    # Connects to DB and grabs EVENT LOGS
    try:
        conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-L3DV0JT;DATABASE=SECAMS;UID=sa;PWD=1')
        sql_query_str = """
WITH USERID_TALLIES (userid, tally)
AS(
SELECT TOP(500) userid, COUNT(*) AS tally
FROM access_event_logs
GROUP BY userid)

SELECT DISTINCT USERID, EVENTID, TERMINALSN, TIMESTAMPS
FROM access_event_logs AS l
WHERE EXISTS (SELECT * 
FROM USERID_TALLIES AS UT
WHERE l.userid = ut.userid
AND uT.tally > 100);"""

        events = pd.read_sql(sql_query_str, conn)
        return events

    except pyodbc.Error:
        return False


def get_events_from_csv(filename="development_data.csv"):
    try:
        events = pd.read_csv(filename)
        # events['TIMESTAMPS'] = events['TIMESTAMPS'].apply(lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S"))
        events['TIMESTAMPS'] = events['TIMESTAMPS'].apply(datetime64)

        # print(type(events['TIMESTAMPS'][0]))
        # print(events['TIMESTAMPS'])

        return events
    except FileNotFoundError:
        return False


def store_to_csv():
    df = get_events_from_sql()
    df.to_csv("development_data.csv")


def get_events():
    # blanket get_event method that tries SQL then CSV

    from_sql = get_events_from_sql()
    if type(from_sql) == pd.DataFrame:
        return from_sql

    from_csv = get_events_from_csv()
    if type(from_csv) == pd.DataFrame:
        return from_csv

    # If neither return, then both didn't work
    exit("ERROR: No data found.")


def get_vocab_lists(column_name):
    # Preset synthetic column names; return an already-defined vocab list
    if column_name == "DAYOFWEEK":
        return ["0", "1", "2", "3", "4", "5", "6"]

    elif column_name == "MONTHOFYEAR":
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    # Connect to DB event_logs table, and grab distinct values (vocab lists), given the column name
    try:
        conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-L3DV0JT;DATABASE=SECAMS;UID=sa;PWD=1')
        query_str = "SELECT " + column_name + " FROM access_event_logs group by " + column_name + ""

        # print(type(pd.read_sql(query_str, conn)['TERMINALSN'][0]))
        return pd.read_sql(query_str, conn).apply(lambda x: str(x))

    except pyodbc.Error:
        exit("ERROR: Cannot find vocab list for " + column_name + "; invalid column name or unable to connect to database.")
        return False


# # testy test test
# get_events()
# store_to_csv()