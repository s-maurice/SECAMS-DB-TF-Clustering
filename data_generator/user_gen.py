import numpy as np
import pandas as pd
import random


def generate_from_user_room_weighting(full_time_weighting_df):
    # Call once with both full and part time, or call twice and generalise?
    week_day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  # List of days, replace with iterweekdays?
    period_list = ["Period"+str(i) for i in range(5)]  # Creates list of periods
    user_df_list = []

    # Check that the room lists and the biases are equal
    # Need to update assertions to assert for all items in the df, not just index 1
    assert (len(full_time_weighting_df["other_rooms"][0]) == len(full_time_weighting_df["other_room_bias"][0])), (
        "Length of other_rooms did not match length of bias")
    assert (len(full_time_weighting_df["lunch_other_room"][0]) == len(full_time_weighting_df["lunch_other_room)_bias"][0])), (
        "Length of lunch_other_rooms did not match length of bias")

    # Define room selector function, which creates a pool of rooms to pick from, and returns a single room
    def room_selector(main_room, main_room_bias, other_rooms, other_room_bias):
        # Create the list of rooms to draw from, weighted by the bias
        # Draws and returns single room
        current_period_room_pool = main_room * main_room_bias
        current_period_room_pool.append([j * k for j, k in zip(other_rooms, other_room_bias)])
        return random.choice(current_period_room_pool)

    for index, row in full_time_weighting_df.itterrows():
        current_user_df = pd.DataFrame()  # Create a fresh Data Frame
        for day in week_day_list:  # 5 days in a week
            for period in period_list:  # 5 periods in a day, excluding lunch
                current_user_df.loc[day, period] = room_selector(
                    row["main_room"],
                    row["main_room_bias"],
                    row["other_rooms"],
                    row["other_room_bias"])
            current_user_df.loc[day, "Lunch"] = room_selector(
                row["lunch_main_room"],
                row["lunch_room_bias"],
                row["lunch_other_rooms"],
                row["lunch_other_room_bias"])

        user_df_list.append(current_user_df)  # Appends completed Data Frame to list of Data Frames

