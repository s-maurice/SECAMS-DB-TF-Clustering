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
def generate_daily_schedule(total_rooms=15,
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

    if num_part_time_users_frac > 0:
        # Randomly picks part time rows according to num_part_time_frac, and replaces classroom values with 0.
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


# New generator
def generator_by_period_remove_classrooms_from_list(lunch_period=3):
    full_time_df = pd.DataFrame()
    part_time_df = pd.DataFrame()
    for i in range(6+1):  # Periods in day, plus one for lunch
        avaliable_classrooms = [("C"+str(i)) for i in range(15)]  # Reset list of avaliable rooms
        used_classrooms = []
        for ii in range(10):  # num of full time teachers
            if i == lunch_period:
                current_room = "L"
            else:
                # Get and remove the current classroom from available classroom list
                current_room_index = np.random.randint(len(avaliable_classrooms))
                current_room = avaliable_classrooms[current_room_index]
                used_classrooms.append(current_room)
                avaliable_classrooms.pop(current_room_index)
            full_time_df.loc["period"+str(i), "full_time"+str(ii)] = current_room  # Add classroom to Data Frame
        for iii in range(5):  # num of part time teachers
            if i == lunch_period:
                current_room = "L"
            else:
                # Adds in bias so that part timers prefer their own classrooms
                # this should be implemented elsewhere, currently has no effect
                current_part_time_bias_list_index = np.random.randint(0, len(avaliable_classrooms), size=2) # biased towards 3 particular classrooms
                current_part_time_bias_list = avaliable_classrooms + 2*[avaliable_classrooms[i] for i in current_part_time_bias_list_index]  # how heavily it's biased

                # Adds in some used classrooms to have teachers double up, biases can be changed as above
                used_classrooms_bias_list_index = np.random.randint(0, len(used_classrooms), size=2)
                used_classrooms_bias_list = [used_classrooms[i] for i in used_classrooms_bias_list_index]
                current_part_time_bias_list += used_classrooms_bias_list  # These two lines can be combined into one

                current_room_index = np.random.randint(len(avaliable_classrooms))
                current_room = current_part_time_bias_list[current_room_index]
            part_time_df.loc["period" + str(i), "part_time" + str(iii)] = current_room  # Add classroom to Data Frame
    # Chop out part time morning/afternoons here


    print(full_time_df)
    print(part_time_df)





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

#
# daily_schedule = generate_daily_schedule()
# print(daily_schedule)
# user_list = generate_user_weekly_schedules(daily_schedule, after_school_meeting=True)
# print(user_list[1])
#generate_timestamps(user_list)

generator_by_period_remove_classrooms_from_list()
