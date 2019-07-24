import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time


# Defines an event based on two booleans; idx is so the method can be applied to a DataFrame
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


# Obtains the reason for a particular event; day in datetime // event as String
# In particular (given Friday/Saturday as the weekend):
#   Absent on a Sunday - Hungover
#   Absent on a Thursday - Skiving
#   Absent on other days - Car broke down / Sick / Other activities
#   Absent on specific days - Holidays
def get_reason(day, event):
    if event == "Absent":
        if day.weekday() == 6:
            return "Hungover"
        elif day.weekday() == 3:
            return "Skiving"
        else:
            return "Sick"
    else:
        return event


def gen_holiday(df, holidate):
    # Generates a holiday; on all 'holidate' events, people are absent due to the reason 'Holiday'
    df.loc[df['Day'] == holidate, 'Present'] = False
    df.loc[df['Day'] == holidate, 'Reason'] = "Holiday"


df = pd.read_csv("absence_df.csv")
df['Day'] = pd.to_datetime(df['Day'])
df.set_index("index", inplace=True)
df['Event'] = [get_event(present, weekday) for present, weekday in zip(df['Present'], df['Weekday'])]
df['Reason'] = [get_reason(day, event) for day, event in zip(df['Day'], df['Event'])]

gen_holiday(df, pd.Timestamp(dt.date(year=2016, month=5, day=18)))

pd.set_option('display.max_rows', 1000)
print(df)

df.to_csv("reason_df.csv")

print('dun')
