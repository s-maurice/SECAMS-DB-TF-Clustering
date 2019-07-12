import random
from random import randint
from random import randrange
import numpy as np
import pandas as pd


# schedule_gen creates a list of DataFrames; each DF indicates the 'schedule' of an individual user.
# Each user created is similar to that of a 'full time' teacher in a school.
# The schedule is defined using 'periods'; during each period, a user is in a 'room'.

# Specific periods have special values:
#   Period 3, the lunch break
#   Period 6, for after-school meetings

# The rooms defined are:
#   C1 through 10, as 'classrooms'
#   B, break (e.g. staffroom)
#   L, lunch (e.g. a canteen)
#   M, meeting room
#   0, away from school (home)

# Each user has the same daily schedule, repeated across weekdays, with the exception of a weekly meeting on Friday.

def generate_daily_schedule():
    # Generates the daily schedule for each user, by creating a room array and shuffling it.
    # This avoids the same classroom being assigned to two teachers during the same period.
    schedule_df = pd.DataFrame()

    # 8 rooms and 2 breaks; during any period, two teachers will be on break.
    # The number of actual teachers in the schedule is equal to len(rooms).
    rooms = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'B', 'B']

    # range(0, 6) for periods 0 through 5; add period 6 later.
    for i in range(0, 6):
        random.shuffle(rooms)
        schedule_df['period' + str(i)] = rooms

    # Reassign rooms during period3, i.e. lunch - note that the length of this must be the same as the length of rooms
    p3rooms = ["B", "C1", "L", "L", "L", "L", "L", "L", "L", "L"]
    random.shuffle(p3rooms)
    schedule_df["period3"] = p3rooms

    return schedule_df


def generate_user_weekly_schedules():
    schedule_df = generate_daily_schedule()

    user_df_list = []
    week_days = ["MON", "TUE", "WED", "THU", "FRI"]
    for index, row in schedule_df.iterrows():
        cur_user = pd.DataFrame()
        for i in range(5):
            cur_user[week_days[i]] = row
        cur_user.loc["period"+ str(len(cur_user.index))] = ["0", "0", "M", "0", "0"]
        user_df_list.append(cur_user)

# print(user_df_list[5])
