import datetime

import numpy as np
import pandas as pd
import random
import scipy


def generate_from_user_room_weighting(full_time_weighting_df, lunch_period=False, end_period_meeting_day="", drop_half=False):
    # Call once with both full and part time, or call twice and generalise?
    week_day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  # List of days, replace with iterweekdays?
    period_list = ["Period" + str(i) for i in range(5)]  # Creates list of periods
    user_df_list = []

    # Check that the room lists and the biases are equal
    # Need to update assertions to assert for all items in the df, not just index 1
    # assert (len(full_time_weighting_df["other_rooms"][0]) == len(full_time_weighting_df["other_room_bias"][0])), (
    #     "Length of other_rooms did not match length of bias")
    # assert (len(full_time_weighting_df["lunch_other_rooms"][0]) == len(full_time_weighting_df["lunch_other_room_bias"][0])), (
    #     "Length of lunch_other_rooms did not match length of bias")

    # Define room selector function, which creates a pool of rooms to pick from, and returns a single room
    def room_selector(main_room, main_room_bias, other_rooms, other_room_bias):
        # Create the list of rooms to draw from, weighted by the bias
        # Draws and returns single room
        current_period_room_pool = [main_room] * main_room_bias
        for k in other_room_bias:
            current_period_room_pool += other_rooms  # Multiplies two lists
        return random.choice(current_period_room_pool)

    for index, row in full_time_weighting_df.iterrows():
        current_user_df = pd.DataFrame()  # Create a fresh Data Frame
        for day in week_day_list:  # 5 days in a week
            for period in period_list:  # 5 periods in a day, excluding lunch
                current_user_df.loc[day, period] = room_selector(
                    row["main_room"],
                    row["main_room_bias"],
                    row["other_rooms"],
                    row["other_room_bias"])
            if lunch_period:
                current_user_df.loc[day, "Lunch"] = room_selector(
                    row["lunch_main_room"],
                    row["lunch_main_room_bias"],
                    row["lunch_other_rooms"],
                    row["lunch_other_room_bias"])
            if day == end_period_meeting_day:
                current_user_df.loc[day, "Normal_End"] = "M"
        if drop_half:  # Removes before/after lunch randomly, used for part timers
            if bool(random.getrandbits(1)):
                current_user_df.iloc[:, 3:] = "O"
            else:
                current_user_df.iloc[:, :3] = "O"

        # Reorder + fill NAs as Os
        current_user_df = current_user_df.reindex(columns=["Period0", "Period1", "Period2", "Lunch", "Period3", "Period4", "Normal_End"])
        current_user_df.fillna("O", inplace=True)

        user_df_list.append(current_user_df)  # Appends completed Data Frame to list of Data Frames
    return user_df_list


def generate_user_df(full_time,
                     part_time,
                     ft_assign_range,
                     pt_assign_range,
                     extra_rooms=0,
                     main_bias_multiplier=1,
                     lunch_bias_multiplier=3,
                     main_room_overlap=False):
    # Generates a DataFrame of users + their characteristics.
    # Args:
    #   full_time: Number of FT teachers
    #   part_time: Number of PT teachers (generally less than FT)
    #   ft_assign_range: Tuple. How many non-unique rooms each FT teacher is assigned
    #   pt_assign_range: Tuple. How many non-unique rooms each PT teacher is assigned (generally less than FT)
    #   extra_rooms: Number of additional 'shared' rooms (if default, then set to the minimum required)
    #   main_bias_multiplier: How many more times likely to choose the 'main room' over any other rooms.
    #   lunch_bias_multiplier: How many more times likely to choose the 'lunch room' over other rooms, during lunch.
    #   main_room_overlap: Not yet implemented. Whether the 'main rooms' of FT teachers overlap or not.

    # NOTE: The 'bias' columns represent 'characteristics' of each user. In particular:
    #   randomness_bias determines how much variation a user exhibits in their activity; DETERMINES OTHER BIASES
    #   time_offset_bias (-1 to 1) refers to generally how early or late a user comes in (uses normal distribution)
    #   absence_bias (0 to 1) represents the chance that the user is absent.
    #   mistake_bias (0 to 1) represents the chance that a user makes a mistake in entry (wrong in/out, buddy punched)
    df_columns = ['user', 'main_room', 'other_rooms', 'lunch_main_room', 'lunch_other_rooms',
                  'main_room_bias', 'other_room_bias', 'lunch_main_room_bias', 'lunch_other_room_bias',
                  'randomness_bias', 'time_offset_bias', 'absence_bias', 'mistake_bias']

    # Hyperparameters on biases. Time_offset changes the centre of the early/late distribution, when calling np.random.normal().
    # 'Flaw' biases (absences, mistakes) are generated using (1 - np.random.power(2)) * limit; ensures that most values are generally lower.
    randomness_bias_limit = 0.5
    time_offset_bias_spread = 0.25
    absence_bias_limit = 0.2
    mistake_bias_limit = 0.2

    def gen_randomness_bias(size):
        return (1 - np.random.power(2, size)) * randomness_bias_limit

    def gen_time_offset_bias(size):
        return np.clip(np.random.normal(loc=0, scale=time_offset_bias_spread, size=size), -1, 1) * randomness_bias_limit

    def gen_time_offset_bias_alt(lower, upper, loc, std, size):
        X = scipy.stats.truncnorm((lower - loc) / std, (upper - loc) / std, loc=loc, scale=std)
        points = X.rvs(size)
        return points

    def gen_absence_bias(size):
        return (1 - np.random.power(8, size)) * absence_bias_limit

    def gen_mistake_bias(size):
        return (1 - np.random.power(8, size)) * mistake_bias_limit

    ft_user_df = pd.DataFrame(columns=df_columns)
    pt_user_df = pd.DataFrame(columns=df_columns)

    # --- GENERATE FT DATAFRAME ---
    # First generate the indexes; 'user' should be a list of full_time and part_time teachers, like so:
    # ['FT0', 'FT1', 'FT2', 'PT0', 'PT1']

    ft_user_list = ["FT" + str(i) for i in range(full_time)]    # ['FT0', 'FT1', ...]
    ft_user_df['user'] = ft_user_list
    ft_user_df.set_index('user', inplace=True)

    # For the main_room column, modify only the main_rooms of FT users, leave PT as NaN
    main_room_list = ["C" + str(i) for i in range(full_time)]   # ['C0', 'C1', ...]
    ft_user_df['main_room'] = main_room_list

    # For other_rooms, find the number of non-unique (shared) rooms
    non_unique_rooms = max(max(ft_assign_range) + extra_rooms, max(pt_assign_range))

    # Create a list of non_unique_rooms; randomly assign a set of them to each full_time user
    # Non-unique rooms have an offset of full_time;
    # e.g. with 5 FT teachers, unique rooms go up to C4, so non-unique begins at C5
    non_unique_room_list = ["C" + str(i) for i in range(full_time, full_time + non_unique_rooms)] # ['C4', 'C5', ...]

    # Create one list for each column; this holds their values. Doing so is much better than
    other_rooms_list = []
    other_room_bias_list = []
    main_room_bias_list = []

    for i in range(full_time):
        # Get rooms for each user
        room_count = random.randint(ft_assign_range[0], ft_assign_range[1])     # how many rooms
        rooms = random.sample(non_unique_room_list, room_count)                 # get rooms by sampling from room list
        rooms.sort()
        other_rooms_list.append(rooms)

        # Non-main room biases
        other_rooms_bias = np.random.randint(1, 3, size=room_count)  # give a preference of anywhere from 1 to 3
        other_room_bias_list.append(other_rooms_bias)

        # Main room bias: Sum of other biases * main_bias_multiplier
        main_room_bias = sum(other_rooms_bias) * main_bias_multiplier
        main_room_bias_list.append(main_room_bias)

    # Assign the lists defined above
    ft_user_df['other_rooms'] = other_rooms_list
    ft_user_df['other_room_bias'] = other_room_bias_list
    ft_user_df['main_room_bias'] = main_room_bias_list

    # Generate other user biases
    ft_user_df['randomness_bias'] = gen_randomness_bias(size=full_time)
    ft_user_df['time_offset_bias'] = gen_time_offset_bias(size=full_time)
    ft_user_df['absence_bias'] = gen_absence_bias(size=full_time)
    ft_user_df['mistake_bias'] = gen_mistake_bias(size=full_time)

    # lunch_main_room: lunch room (L); set bias to lunch main room bias
    ft_user_df['lunch_main_room'] = "L"
    ft_user_df['lunch_main_room_bias'] = lunch_bias_multiplier

    # lunch_other_rooms: main_room, and break room (B); hard-code bias of 1, 1 in for now
    ft_user_df['lunch_other_rooms'] = ft_user_df['main_room'].apply(lambda x: [x] + ["B"])
    ft_user_df["lunch_other_room_bias"] = [[1, 1]] * full_time

    # --- GENERATE PT DATAFRAME ---
    # Generate indexes
    pt_user_list = [("PT" + str(i)) for i in range(part_time)]  # append ['PT0', 'PT1', ...]
    pt_user_df['user'] = pt_user_list
    pt_user_df.set_index('user', inplace=True)

    # For other_rooms, take the list of non_unique_rooms; randomly assign a set of them to each part_time user
    other_rooms_list = []
    other_room_bias_list = []

    for i in range(part_time):
        # Get rooms for each user
        room_count = random.randint(pt_assign_range[0], pt_assign_range[1])     # how many rooms
        rooms = random.sample(non_unique_room_list, room_count)                 # get rooms by sampling from room list
        rooms.sort()
        other_rooms_list.append(rooms)

        # Other room biases
        other_rooms_bias = np.random.randint(1, 3, size=room_count)  # give a preference of anywhere from 1 to 3
        other_room_bias_list.append(other_rooms_bias)

    # Assign the lists defined above
    pt_user_df['other_rooms'] = other_rooms_list
    pt_user_df['other_room_bias'] = other_room_bias_list

    # Generate other user biases
    pt_user_df['randomness_bias'] = gen_randomness_bias(size=part_time)
    pt_user_df['time_offset_bias'] = gen_time_offset_bias(size=part_time)
    pt_user_df['absence_bias'] = gen_absence_bias(size=part_time)
    pt_user_df['mistake_bias'] = gen_mistake_bias(size=part_time)

    # Turn all NaN biases into 0s
    pt_user_df.fillna(0, inplace=True)

    return ft_user_df, pt_user_df


def generate_event_list(schedule_df_list, bias_df, num_weeks):

    def generate_day_sched(day_series, day, week):
        day_sched_df = pd.DataFrame(columns=["Week", "Day", "Room", "Time", "Event"])

        room_list = []
        time_list = []
        event_list = []
        for i in range(len(day_series)):
            if day_series[i] != "O":
                if (i - 1 < 0) or (i - 1 >= 0 and day_series.iloc[i-1] != day_series.iloc[i]):
                    event_list.append('In')
                    room_list.append(day_series[i])
                    time_list.append(day_series.index[i])
                if (i + 1 >= len(day_series)) or (i + 1 < len(day_series) and day_series.iloc[i+1] != day_series.iloc[i]):
                    event_list.append('Out')
                    room_list.append(day_series[i])
                    time_list.append(day_series.index[i])

        day_sched_df['Room'] = room_list
        day_sched_df['Time'] = time_list
        day_sched_df['Event'] = event_list
        # Set 'Week' and 'Day' columns only after setting other columns.
        day_sched_df['Week'] = week
        day_sched_df['Day'] = day

        # Add entering and exiting school events, doesn't currently work for part timers
        # School entry event timing
        if time_list[0] == "Period0":  # Check if they come in at normal time
            school_entry_time = "Normal_Start"
        elif time_list[0] == "Period3":  # Check if they come in after lunch
            school_entry_time = "Late_Start"
        else:  # Catch Errors
            school_entry_time = "Error_Start"
            print("error, start not found, Error_Start appended, first time_List value was: " + time_list[0])

        # School exit event timing
        if time_list[-1] == "Normal_End":  # Check if they have after school meeting, if so shift time
            school_exit_time = "Late_End"
        elif time_list[-1] == "Period2":  # Check if they end school at lunch
            school_exit_time = "Early_End"
        elif time_list[-1] == "Period4":  # Check if they end normally
            school_exit_time = "Normal_End"
        else:  # Catch errors
            school_exit_time = "Error_End"
            print("error, ending not found, Error_End appended, final time_list value was: " + time_list[-1])

        day_sched_df.loc[-1] = [week, day, "Main Gate", school_entry_time, "In"]  # Enter School, appends to bottom
        day_sched_df.index = day_sched_df.index + 1  # Shifts index by 1
        day_sched_df.sort_index(inplace=True)  # Sorts so that school entry is at top
        day_sched_df.loc[-1] = [week, day, "Main Gate", school_exit_time, "Out"]  # Exit School

        day_sched_df.set_index("Time", drop=True, inplace=True)
        return day_sched_df

    # Define a list to hold the weekly schedule DFs; one for each user
    user_event_df_list = []

    for user_df_index, user_df in enumerate(schedule_df_list):
        # Convert Schedule into list of events
        user_event_df = pd.DataFrame()
        for week in range(num_weeks):
            for index, row in user_df.iterrows():
                # Check if user is absent before creating and appending the current day to their event list
                if bias_df["absence_bias"][user_df_index] < np.random.random():
                    cur_user_event_day_df = generate_day_sched(row, index, week)
                    user_event_df = user_event_df.append(cur_user_event_day_df)

            # Add how early/late each event is as a new column, "Early/Lateness"
            user_event_df["Early/Lateness"] = np.random.normal(loc=bias_df["time_offset_bias"][user_df_index],
                                                               scale=bias_df["randomness_bias"][user_df_index],
                                                               size=len(user_event_df))

        # Add user_df_index to column for easy identification of user_ids
        user_event_df["UserID"] = user_df_index

        user_event_df_list.append(user_event_df)

    return user_event_df_list


def generate_timestamps(user_event_df_list, start_datetime):  # Generates timestamps and event_logs from the list of user_event_dfs
    # Create dict for period to datetime lookup
    period_to_timestamp_dict = {"Normal_Start": datetime.timedelta(hours=8, minutes=30),
                                "Period0": datetime.timedelta(hours=9),
                                "Period1": datetime.timedelta(hours=10),
                                "Period2": datetime.timedelta(hours=11),
                                "Early_End": datetime.timedelta(hours=11, minutes=30),
                                "Lunch": datetime.timedelta(hours=12),
                                "Late_Start": datetime.timedelta(hours=12, minutes=30),
                                "Period3": datetime.timedelta(hours=13),
                                "Period4": datetime.timedelta(hours=14),
                                "Normal_End": datetime.timedelta(hours=14, minutes=10),
                                "Late_End": datetime.timedelta(hours=15, minutes=10)}
    complete_event_df = pd.concat(user_event_df_list)  # First combine all the DFs from the list

    complete_event_df.index = complete_event_df.index.map(period_to_timestamp_dict)  # Applies dict

    # # Convert Early/Lateness column into timedelta, 1 = 30 minutes for now
    # complete_event_df["Early/Lateness"] = complete_event_df["Early/Lateness"].apply(lambda x: datetime.timedelta(minutes=(x * 30)))

    # Convert to complete datetime timestamp
    def create_complete_datetime_timestamps(row):
        week_day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]  # Create a list of weeks for index lookup
        current_timestamp = start_datetime + row.name  # Add the start_timestamp offset
        current_timestamp += datetime.timedelta(weeks=row["Week"], days=week_day_list.index(row["Day"]))  # Add the week and day timedelta
        current_timestamp += datetime.timedelta(minutes=row["Early/Lateness"] * 30)



        # if 'OUT' event or meeting event, add 50 minutes
        if row["Event"] == "Out":
            current_timestamp += datetime.timedelta(minutes=50)
        if row["Room"] == "M":
            current_timestamp += datetime.timedelta(minutes=50)

        return current_timestamp

    # Applies created func
    complete_event_df["Timestamps"] = complete_event_df[["Day", "Week", "Early/Lateness", "Event", "Room"]].apply(create_complete_datetime_timestamps, axis=1)

    # Drop other time columns
    complete_event_df.drop(["Week", "Day", "Early/Lateness"], axis=1, inplace=True)
    complete_event_df.reset_index(drop=True, inplace=True)

    return complete_event_df



pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10)

# First generate a user DataFrame, creating characteristics of Full and Part-time staff
ft_bias_list, pt_bias_list = generate_user_df(full_time=4,
                                              part_time=3,
                                              ft_assign_range=(3, 4),
                                              pt_assign_range=(4, 5),
                                              extra_rooms=2,
                                              main_bias_multiplier=2)

# From the user characteristics, generate a list of DataFrames that hold each user's weekly schedule
ft_week_sched_list = generate_from_user_room_weighting(ft_bias_list, lunch_period=True, end_period_meeting_day="Wednesday")
pt_week_sched_list = generate_from_user_room_weighting(pt_bias_list, drop_half=True)

# From the weekly schedules of each user, generate a DF holding all their usual logging events (with respect to biases)
ft_event_df_list = generate_event_list(ft_week_sched_list, ft_bias_list, num_weeks=2)
pt_event_df_list = generate_event_list(pt_week_sched_list, pt_bias_list, num_weeks=2)

# From the list of DFs containing each user's list of events, generate a DF with all the timestamps of each event
ft_event_log_df = generate_timestamps(ft_event_df_list, datetime.datetime(2019, 7, 1))
pt_event_log_df = generate_timestamps(ft_event_df_list, datetime.datetime(2019, 7, 1))

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10)

# Print Function
print_user_type = "ft"
print_index = slice(0,-1)
to_print = True

if to_print:
    if print_user_type == "ft":
        bias_list_print = ft_bias_list.iloc[print_index]
        week_sched_list_print = ft_week_sched_list[print_index]
        event_list_print = ft_event_df_list[print_index]
        event_log_df = ft_event_log_df
    elif print_user_type == "pt":
        bias_list_print = pt_bias_list.iloc[print_index]
        week_sched_list_print = pt_week_sched_list[print_index]
        event_list_print = pt_event_df_list[print_index]
        event_log_df = pt_event_log_df

    print("Relevant user: ")
    print(bias_list_print)
    print("User Schedule:")
    print(week_sched_list_print)
    print("User eventlist:")
    print(event_list_print)
    print(event_log_df)


# Debugging
def to_timedeltas(ft_event_log_df):
    for userid in ft_event_log_df['UserID'].unique():
        user_df = ft_event_log_df[ft_event_log_df['UserID'] == userid]
        user_df.reset_index(drop=True, inplace=True)
        user_df['Time_delta'] = datetime.timedelta(seconds=0)

        for index, row in user_df.iterrows():
            if index >= 1:
                user_df['Time_delta'][index] = row['Timestamps'] - user_df['Timestamps'][index-1]
        print(user_df)

to_timedeltas(ft_event_log_df)