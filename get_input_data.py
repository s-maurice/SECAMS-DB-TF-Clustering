import pyodbc
import pandas as pd


def get_events():  # Replace pyodbe with pandas built in sql
    # Connects to DB and grabs EVENT LOGS
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


def get_events_from_csv():
    events = pd.read_csv("development_data.csv")
    return events


def store_to_csv():
    df = get_events()
    df.to_csv("development_data.csv")
