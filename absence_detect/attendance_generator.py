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
        return "Weekend_Work"
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


def gen_leave(df, leave_threshold=2):
    # Generate 'leave' reasons (as applied leave), with a threshold
    start_time = time.time()
    for idx, row in df.iterrows():
        # Check that index doesn't overflow OR Check if prev_abs of next is lower
        if idx + 1 == len(df.index) or df.loc[idx, 'Prev_absences'] > df.loc[idx + 1, 'Prev_absences']:
            # Check if prev_abs hits leave_threshold
            if df.loc[idx, 'Prev_absences'] >= leave_threshold:
                # Set the last few lines to 'Leave', if a weekday
                for i in range(df['Prev_absences'][idx] + 1):
                    if df.loc[idx-i, 'Day'].weekday() != 4 and df.loc[idx-i, 'Day'].weekday() != 5:
                        df.loc[idx-i, 'Reason'] = 'Leave'
        if idx % 5000 == 0:
            time_taken = time.time() - start_time
            print(str(idx) + ' reached ; ' + str(time_taken) + ' seconds')
            start_time = time.time()

df = pd.read_csv("absence_df.csv")
df['Day'] = pd.to_datetime(df['Day'])
df.set_index("index", inplace=True)
df['Event'] = [get_event(present, weekday) for present, weekday in zip(df['Present'], df['Weekday'])]
df['Reason'] = [get_reason(day, event) for day, event in zip(df['Day'], df['Event'])]

gen_leave(df)
gen_holiday(df, pd.Timestamp(dt.date(year=2016, month=5, day=18)))

pd.set_option('display.max_rows', 1000)
print(df)

df.to_csv("reason_df.csv")

print('dun')
