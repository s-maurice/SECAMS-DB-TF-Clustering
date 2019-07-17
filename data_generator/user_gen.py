import numpy as np
import pandas as pd
import random


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
                current_user_df.loc[day, "End"] = "M"
        if drop_half:  # Removes before/after lunch randomly, used for part timers
            if bool(random.getrandbits(1)):
                current_user_df.iloc[:, 3:] = 0
            else:
                current_user_df.iloc[:, :3] = 0

        # Reorder + fill NAs as 0
        current_user_df = current_user_df.reindex(columns=["Period0", "Period1", "Period2", "Lunch", "Period3", "Period4", "End"])
        current_user_df.fillna("0", inplace=True)

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
    # Generates a DataFrame of users.
    # Args:
    #   full_time: Number of FT teachers
    #   part_time: Number of PT teachers (generally less than FT)
    #   ft_assign_range: Tuple. How many non-unique rooms each FT teacher is assigned
    #   pt_assign_range: Tuple. How many non-unique rooms each PT teacher is assigned (generally less than FT)
    #   extra_rooms: Number of additional 'shared' rooms (the number of 'shared' rooms is usually set to the minimum required)
    #   main_bias_multiplier: How many more times likely to choose the 'main room' over any other rooms.
    #   lunch_bias_multiplier: How many more times likely to choose the 'lunch room' over other rooms, during lunch.
    #   main_room_overlap: Not yet implemented. Whether the 'main rooms' of FT teachers overlap or not.

    ft_user_df = pd.DataFrame(columns=['user', 'main_room', 'other_rooms', 'lunch_main_room', 'lunch_other_rooms', 'main_room_bias', 'other_room_bias', 'lunch_main_room_bias', 'lunch_other_room_bias'])
    pt_user_df = pd.DataFrame(columns=['user', 'main_room', 'other_rooms', 'lunch_main_room', 'lunch_other_rooms', 'main_room_bias', 'other_room_bias', 'lunch_main_room_bias', 'lunch_other_room_bias'])

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
    # Give an offset of full_time; e.g. if there are 5 FT teachers, unique rooms go up to C4, so begin non_unique rooms at C5
    non_unique_room_list = ["C" + str(i) for i in range(full_time, full_time + non_unique_rooms)] # ['C4', 'C5', ...]
    other_rooms_list = []
    other_room_bias_list = []
    main_room_bias_list = []
    for i in range(full_time):
        # Get rooms for each user
        room_count = random.randint(ft_assign_range[0], ft_assign_range[1])     # how many rooms
        rooms = random.sample(non_unique_room_list, room_count)                 # get rooms by sampling from room list
        rooms.sort()
        other_rooms_list.append(rooms)

        # Get bias for each user
        # Non-main rooms
        other_rooms_bias = np.random.randint(1, 3, size=room_count)  # give a preference of anywhere from 1 to 3
        other_room_bias_list.append(other_rooms_bias)

        # Main room: Sum of other biases * main_bias_multiplier
        main_room_bias = sum(other_rooms_bias) * main_bias_multiplier
        main_room_bias_list.append(main_room_bias)

    ft_user_df['other_rooms'] = other_rooms_list
    ft_user_df['other_room_bias'] = other_room_bias_list
    ft_user_df['main_room_bias'] = main_room_bias_list

    # lunch_main_room: lunch room (L); set bias to lunch main room bias
    ft_user_df['lunch_main_room'] = "L"
    ft_user_df['lunch_main_room_bias'] = lunch_bias_multiplier

    # lunch_other_rooms: main_room, and break room (B); hard-code bias of 1, 1 in for now
    ft_user_df['lunch_other_rooms'] = ft_user_df['main_room'].apply(lambda x: [x] + ["B"])
    ft_user_df["lunch_other_room_bias"] = [[1, 1]] * full_time


    # print(ft_user_df)
    # return ft_user_df

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

        # Get bias for each user
        # Non-main rooms
        other_rooms_bias = np.random.randint(1, 3, size=room_count)  # give a preference of anywhere from 1 to 3
        other_room_bias_list.append(other_rooms_bias)

    pt_user_df['other_rooms'] = other_rooms_list
    pt_user_df['other_room_bias'] = other_room_bias_list

    # Turn all NaNs into 0s
    pt_user_df.fillna(0, inplace=True)

    return ft_user_df, pt_user_df


def generate_event_list(schedule_df_list, num_weeks, bias_df):

    def generate_day_sched(day_series, day, week):
        day_sched_df = pd.DataFrame(columns=["Week", "Day", "Room", "Time", "Event"])

        room_list = []
        time_list = []
        event_list = []
        for i in range(len(day_series)):
            if day_series[i] != "0":
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
        day_sched_df['Week'] = week
        day_sched_df['Day'] = day

        # Add entering and exiting school events
        if time_list[-1] == "End":  # Check if they have after school meeting, if so shift time
            school_exit_time = "Late End"
        else:
            school_exit_time = "End"

        day_sched_df.loc[0] = [week, day, "Main Gate", "Start", "In"]  # Enter School
        day_sched_df.loc[-1] = [week, day, "Main Gate", school_exit_time, "Out"]  # Exit School

        day_sched_df.set_index("Time", drop=True, inplace=True)
        return day_sched_df

    user_event_df_list = []

    for user_df_index, user_df in enumerate(schedule_df_list):
        # Convert Schedule into list of events
        user_event_df = pd.DataFrame()
        for week in range(num_weeks):
            for index, row in user_df.iterrows():  # every entry: week - day - room - time - event
                cur_user_event_day_df = generate_day_sched(row, index, week)
                # Check if user is absent before appending the current day to their event list
                if row.loc["absence_bias"] < np.random.random:
                    user_event_df = user_event_df.append(cur_user_event_day_df)

            # Add how early/late each event is as a new column, "Early/Lateness"
            for index, row in user_event_df.iterrows():
                user_event_df.loc[index, "Early/Lateness"] = np.clip(np.random.normal(loc=bias_df.loc["time_offset_bias", user_df_index], scale=1), -1, 1)

        user_event_df_list.append(user_event_df)

    return user_event_df_list


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
ft_sched_df, pt_sched_df = generate_user_df(full_time=4,
                                            part_time=3,
                                            ft_assign_range=(3, 4),
                                            pt_assign_range=(4, 5),
                                            extra_rooms=2,
                                            main_bias_multiplier=2)

ft_list = generate_from_user_room_weighting(ft_sched_df, lunch_period=True, end_period_meeting_day="Wednesday")
pt_list = generate_from_user_room_weighting(pt_sched_df, drop_half=True)
#
# # Test prints
# print("Full-timers: ")
# for df in ft_list:
#     print(df)
#     print()
#
# print("Part-timers: ")
# for df in pt_list:
#     print(df)
#     print()

gay_df_list = generate_event_list(ft_list, 2, ft_sched_df)

print(gay_df_list[2])