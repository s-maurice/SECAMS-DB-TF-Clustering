import pyodbc
import pandas as pd


def get_events_from_sql():  # Replace pyodbe with pandas built in sql
    # Connects to DB and grabs EVENT LOGS
    try:
        conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-L3DV0JT;DATABASE=SECAMS;UID=sa;PWD=1')
        sql_query_str = """
    SELECT TOP(1000) USERID, EVENTID, TIMESTAMPS, access_event_logs.TERMINALSN, TERMINALGROUP, TERMINALNAME
    FROM access_event_logs, access_terminal
    WHERE 1 = CASE
    WHEN ISNUMERIC(USERID) = 1 THEN CASE WHEN CAST(USERID as BIGINT) < 1000000 THEN 1 END
    END
    AND 
    TIMESTAMPS > '2003-1-1'
    AND 
    access_event_logs.TerminalSN = access_terminal.TerminalSN
    ORDER BY USERID"""

        events = pd.read_sql(sql_query_str, conn)
        return events

    except pyodbc.Error:
        return False


def get_events_from_csv():
    try:
        events = pd.read_csv("development_data.csv")
        return events
    except FileNotFoundError:
        return False


def store_to_csv():
    df = get_events_from_sql()
    df.to_csv("development_data.csv")


def get_events():
    # blanket get_event method that tries both SQL and CSV

    from_sql = get_events_from_sql()
    if type(from_sql) == pd.DataFrame:
        return from_sql

    from_csv = get_events_from_csv()
    if type(from_csv) == pd.DataFrame:
        return from_csv

    # If neither return, then both didn't work
    exit("ERROR: No data found.")


# testy test test
# get_events()