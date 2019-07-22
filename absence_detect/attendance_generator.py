import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time


# Defines an event based on two booleans
def get_event(present, weekday):
    # Normal: present on weekdays, absent on weekends
    if present == weekday:
        return "Normal"
    # Present on a weekend
    elif present:
        return "Weekend"
    # Absent on a weekday
    elif not present:
        return "Absent"


# Defines an event based on two booleans
def get_event_alt(x):
    present = x[0]
    weekday = x[1]
    # Normal: present on weekdays, absent on weekends
    if present == weekday:
        return "Normal"
    # Present on a weekend
    elif present:
        return "Weekend"
    # Absent on a weekday
    elif not present:
        return "Absent"


df = pd.read_csv("absence_df.csv")
df.set_index("index", inplace=True)
df['Event'] = df[['Present', 'Weekday']].apply(get_event, args=(1,), axis=1)

print(df)