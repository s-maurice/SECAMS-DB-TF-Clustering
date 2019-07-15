import random
from random import randint
from sklearn import model_selection
from random import randrange
import math
import numpy as np
import pandas as pd
import datetime


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
def generate_daily_schedule(total_rooms=20,
                            periods=8,
                            break_rooms=2,
                            lunch_period_position=3,
                            lunch_break_rooms=1,
                            period_offset=0,
                            num_part_time_users_frac=0.3,
                            ):
    random.seed()  # Without this, random isn't very random
    # Generates the daily schedule for each user, by creating a room array and shuffling it.
    # This avoids the same classroom being assigned to two teachers during the same period.
    schedule_df = pd.DataFrame()

    # Creates room list
    rooms = break_rooms * ["B"]
    [rooms.append("C"+str(i)) for i in range(total_rooms - break_rooms)]

    # Create periods, starting at period0 and ending at period(periods-1)
    for i in range(periods):
        random.shuffle(rooms)
        schedule_df['period' + str(i+period_offset)] = rooms

    # Create list of rooms to be used at lunchtime, list length must equal the length of room list above
    lunch_rooms = (total_rooms - lunch_break_rooms) * ["L"]
    [lunch_rooms.append("C"+str(i)) for i in range(lunch_break_rooms)]

    # Shuffles and adds lunch period to df
    random.shuffle(lunch_rooms)
    schedule_df["period" + str(lunch_period_position)] = lunch_rooms

    # Randomly picks part time rows according to num_part_time_frac, and replaces classroom values with 0,
    # leaving only part_time_periods number of periods remaining. Part time workers work mornings if
    # shuffle_part_time_periods is true, otherwise their start and end periods are randomised.
    # Random may also select same part time to apply to, so number of part timers may not always be the same.

    if num_part_time_users_frac > 0:
        part_time_rows_df = schedule_df.sample(math.floor(num_part_time_users_frac * total_rooms))  # Gets random rows, these will be the part time workers
        before_lunch = part_time_rows_df.sample(randrange(len(part_time_rows_df)+1))  # Splits the dataframe
        after_lunch = part_time_rows_df.drop(before_lunch.index)  # Gets the other side of the split

        before_lunch.iloc[:, lunch_period_position:] = 0   # They don't eat lunch after morning work
        after_lunch.iloc[:, :lunch_period_position+1] = 0  # They come after lunch
        # In order to randomise lunch eating behaviour, need to remake this into loops,
        # iterating through individually instead of just bulk selecting with iloc

        schedule_df.update(before_lunch)
        schedule_df.update(after_lunch)
    return schedule_df


# Each user has the same daily schedule, repeated across weekdays, with the exception of a weekly meeting on Wednesday.
# Turns daily schedule df into a list of DataFrames,
# with each dataframe representing a single user's weekly schedule,
# with each column representing a day of the week from Monday to Friday and each row representing a period.
def generate_user_weekly_schedules(schedule_df,
                                   after_school_meeting=False
                                   ):
    user_df_list = []
    week_days = ["MON", "TUE", "WED", "THU", "FRI"]

    for index, row in schedule_df.iterrows():
        cur_user = pd.DataFrame()  # Generates new DataFrame for the current user
        for i in range(5):
            cur_user[week_days[i]] = row
        if after_school_meeting:
            cur_user.loc["period" + str(len(cur_user.index))] = ["0", "0", "M", "0", "0"]  # After school events hard coded, in this case Wednesday is a meeting day
        user_df_list.append(cur_user)  # Adds completed current user DataFrame to the list of DataFrames

    return user_df_list


def generate_timestamps(user_list):
    for i in user_list:  # Iterate through users
        # First generate user biases
        lateness_bias = 0
        level_of_randomness_bias = 0
        absence_bias = 0

        a = datetime.datetime(2011, 1, 1, 4, 5, 6)
    print(a)
    print(user_list[0])


daily_schedule = generate_daily_schedule()
#print(daily_schedule)
user_list = generate_user_weekly_schedules(daily_schedule, after_school_meeting=True)
generate_timestamps(user_list)
