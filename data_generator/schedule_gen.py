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

def generate_daily_schedule(total_rooms=10, break_rooms=2, lunch_period_position=3, lunch_break_rooms=1, periods=8):
    # Generates the daily schedule for each user, by creating a room array and shuffling it.
    # This avoids the same classroom being assigned to two teachers during the same period.
    schedule_df = pd.DataFrame()

    # Creates room list
    rooms = break_rooms * ["B"]
    [rooms.append("C"+str(i)) for i in range(total_rooms - break_rooms)]

    # Create periods, starting at period0 and ending at period(periods-1)
    for i in range(periods):
        random.shuffle(rooms)
        schedule_df['period' + str(i)] = rooms

    # Create list of rooms to be used at lunchtime, list length must equal the length of room list above
    lunch_rooms = (total_rooms - lunch_break_rooms) * ["L"]
    [lunch_rooms.append("C"+str(i)) for i in range(lunch_break_rooms)]

    # Shuffles and adds lunch period to df
    random.shuffle(lunch_rooms)
    schedule_df["period" + str(lunch_period_position)] = lunch_rooms

    return schedule_df


def generate_user_weekly_schedules(schedule_df=generate_daily_schedule()):
    # Turns daily schedule df into a list of DataFrames, with each dataframe representing a single user's weekly schedule,
    # with each column representing a day of the week from Monday to Friday and each row representing a period.
    user_df_list = []
    week_days = ["MON", "TUE", "WED", "THU", "FRI"]

    for index, row in schedule_df.iterrows():
        cur_user = pd.DataFrame()  # Generates new DataFrame for the current user
        for i in range(5):
            cur_user[week_days[i]] = row
        cur_user.loc["period" + str(len(cur_user.index))] = ["0", "0", "M", "0", "0"]  # After school events hard coded, in this case Wednesday is a meeting day
        user_df_list.append(cur_user)  # Adds completed current user DataFrame to the list of DataFrames

    return user_df_list


a = generate_user_weekly_schedules()
print(a[5])
