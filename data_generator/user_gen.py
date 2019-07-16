import numpy as np
import pandas as pd
import random


def generate_from_user_room_weighting(full_time_weighting_df, lunch_period=False, end_period_meeting_day=""):
    # Call once with both full and part time, or call twice and generalise?
    week_day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  # List of days, replace with iterweekdays?
    period_list = ["Period"+str(i) for i in range(5)]  # Creates list of periods
    user_df_list = []

    # Check that the room lists and the biases are equal
    # Need to update assertions to assert for all items in the df, not just index 1
    assert (len(full_time_weighting_df["other_rooms"][0]) == len(full_time_weighting_df["other_room_bias"][0])), (
        "Length of other_rooms did not match length of bias")
    assert (len(full_time_weighting_df["lunch_other_rooms"][0]) == len(full_time_weighting_df["lunch_other_room_bias"][0])), (
        "Length of lunch_other_rooms did not match length of bias")

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

        current_user_df.fillna(0, inplace=True)  # FIlls NAs as 0
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

    # Create the number of non-unique (shared) rooms
    non_unique_rooms = max(ft_assign_range + pt_assign_range) + extra_rooms

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

    # For the other_rooms, create a list of non_unique_rooms; randomly assign a set of them to each full_time user
    # Give an offset of full_time; e.g. if there are 5 FT teachers, unique rooms go up to C4, so begin non_unique rooms at C5
    non_unique_room_list = ["C" + str(i) for i in range(full_time, full_time + non_unique_rooms)] # ['C4', 'C5', ...]
    other_room_list = []
    other_room_bias_list = []
    main_room_bias_list = []
    for i in range(full_time):
        # Get rooms for each user
        room_count = random.randint(ft_assign_range[0], ft_assign_range[1])     # how many rooms
        rooms = random.sample(non_unique_room_list, room_count)                 # get rooms by sampling from room list
        rooms.sort()
        other_room_list.append(rooms)

        # Get bias for each user
        # Non-main rooms
        other_rooms_bias = np.random.randint(1, 3, size=room_count)  # give a preference of anywhere from 1 to 3
        other_room_bias_list.append(other_rooms_bias)

        # Main room: Sum of other biases * main_bias_multiplier
        main_room_bias = sum(other_rooms_bias) * main_bias_multiplier
        main_room_bias_list.append(main_room_bias)

    ft_user_df['other_rooms'] = other_room_list
    ft_user_df['other_room_bias'] = other_room_bias_list
    ft_user_df['main_room_bias'] = main_room_bias_list

    # lunch_main_room: lunch room (L); set bias to lunch main room bias
    ft_user_df['lunch_main_room'] = "L"
    ft_user_df['lunch_main_room_bias'] = lunch_bias_multiplier

    # lunch_other_rooms: main_room, and break room (B); hard-code bias of 1, 1 in for now
    ft_user_df['lunch_other_rooms'] = ft_user_df['main_room'].apply(lambda x: [x] + ["B"])
    ft_user_df["lunch_other_room_bias"] = [[1, 1]] * full_time

    return ft_user_df

    # --- GENERATE PT DATAFRAME ---
    pt_user_list = [("PT" + str(i)) for i in range(part_time)]  # append ['PT0', 'PT1', ...]
    pt_user_df['user'] = pt_user_list
    pt_user_df.set_index('user', inplace=True)

    # print(pt_user_df)


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
gay_df = generate_user_df(full_time=4,
                          part_time=3,
                          ft_assign_range=(3, 4),
                          pt_assign_range=(1, 2),
                          extra_rooms=2,
                          main_bias_multiplier=2)

a = generate_from_user_room_weighting(gay_df, lunch_period=True, end_period_meeting_day="Wednesday")
print(a[2])
