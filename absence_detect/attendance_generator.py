import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
from scipy.stats import truncnorm


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


def random(sick_weight=0,
           car_weight=0,
           train_weight=0,
           skiving_weight=0,
           hungover_weight=0,
           other_weight=0,
           rare_weight=0):
    # Method that chooses a random item from 'choice' depending on the weights.
    # Uses np.random.choice(), but allows weights to add up to any total value.

    events = ["Sick", "Car Broke", "Train Failure", "Skiving", "Hungover"]    # Specific weights
    other_events = ["Dentist", "Court Appointment", "Meeting"]                # 'Other' weights
    rare_events = ["Funeral", "Hospital"]                                     # 'Rare' weights

    weights = [sick_weight, car_weight, train_weight, skiving_weight, hungover_weight] + [other_weight] * len(other_events) + [rare_weight] * len(rare_events)

    all_events = events + other_events + rare_events
    normalised_weights = [weight / sum(weights) for weight in weights]

    return np.random.choice(all_events, p=normalised_weights)


# Obtains the reason for a particular event; day in datetime // event as String
# In particular (given Friday/Saturday as the weekend):
#   Absent on a Sunday - Sick / Car / Hungover
#   Absent on a Thursday - Sick / Car / Skiving
#   Absent on other days - Sick / Car
#   Absent over multiple days - (Applied) Leave
#   Absent on specific days - Holidays
def get_reason(day, event, userid):

    if event == "Absent":
        # Use the userid to get a specific random, user-specific bias that decides other biases (between -1 and 1)
        user_bias = (math.floor(abs(hash(str(userid)) % 10**3)) / 10**3)*2 - 1

        sick_weight = 0
        car_weight = 0
        train_weight = 0
        skiving_weight = 0
        hungover_weight = 0
        other_weight = 0
        rare_weight = 0

        # Define some weights straight off user_bias
        sick_weight = 0.5 + user_bias*0.2
        other_weight = 0.05 + user_bias*0.05
        rare_weight = 0.01 + user_bias*0.01

        # Transportation weights - will either only use car or train
        if abs(hash(str(userid))) % 2 == 0:
            train_weight = 0.05 + user_bias*0.05
        else:
            car_weight = 0.05 + user_bias*0.05

        # Skiving/Hungover weights - only if absent, and depending on the day of the week
        week_edge_bias = 0.1 + user_bias*0.1
        if day.weekday() == 6:  # absent on first day of week
            hungover_weight = week_edge_bias
        elif day.weekday() == 3:   # absent on last day of week
            skiving_weight = week_edge_bias

        return random(sick_weight=sick_weight,
                      car_weight=car_weight,
                      train_weight=train_weight,
                      skiving_weight=skiving_weight,
                      hungover_weight=hungover_weight,
                      other_weight=other_weight,
                      rare_weight=rare_weight)

    # If not absent, then return the event as is
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
df['Reason'] = [get_reason(day, event, userid) for day, event, userid in zip(df['Day'], df['Event'], df['USERID'])]

gen_leave(df)
gen_holiday(df, pd.Timestamp(dt.date(year=2016, month=5, day=18)))

pd.set_option('display.max_rows', 1000)
print(df)

df.to_csv("reason_df.csv")

print('dun')
