import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time

def main():

    def is_weekday(date):
        # Weekdays: Sunday to Thursday, i.e. 0, 1, 2, 3, 6 // Weekends are 4, 5
        return (not date.weekday() == 4) and (not date.weekday() == 5)

    # Read and prepare DF
    df = pd.read_csv("user_day_entry_30to70.csv")
    df['DOY'] = [dt.date(year=2016, month=month, day=day) for month, day in zip(df['e_month'], df['e_day'])]

    # Date range
    start_date = dt.date(year=2016, month=5, day=1)
    end_date = dt.date(year=2016, month=7, day=1)

    # Only include data points inside date range
    df = df[df['DOY'] > start_date]
    df = df[df['DOY'] < end_date]

    # Create in_days and in_days_series - a list/Series of all days in the date range
    in_days = []
    for date in (start_date + dt.timedelta(days=n) for n in range((end_date - start_date).days)):
        in_days.append(date)
    in_days_series = pd.Series(in_days)

    # absence_df - DataFrame containing every single absence and presence of each user
    absence_df = pd.DataFrame(columns=['USERID', 'Day', 'Present', 'Weekday', 'Prev_absences'])

    # lists to hold each column
    userid_column = []
    # day_column = []
    present_column = []
    # weekday_column = []
    prev_abs_column = []

    i = 1

    for userid in df['USERID'].unique():
        user_df = df.loc[df['USERID'] == userid]    # slice of DataFrame only containing one user
        isin_list = in_days_series.isin(user_df['DOY'])    # List of true/false per day on whether user was in

        # Look at previous absences (for use as a synthetic feature)
        prev_abs_list = []
        for index in range(len(isin_list)):
            if index == 0:
                prev_abs_list.append(0)
            elif isin_list[index] == True or isin_list[index-1] == True:
                prev_abs_list.append(0)
            else:
                prev_abs_list.append(prev_abs_list[index-1] + 1)

        userid_column += [userid] * len(isin_list)
        # day_column.extend()
        present_column += isin_list.tolist()
        # weekday_column.extend()
        prev_abs_column += prev_abs_list

        if i % 400 == 0:
            print(str(i) + ' users reached')
        i += 1

    # Flatten lists
    absence_df['USERID'] = userid_column
    absence_df['Present'] = present_column
    absence_df['Prev_absences'] = prev_abs_column

    absence_df['Day'] = in_days * len(df['USERID'].unique())
    absence_df['Weekday'] = [is_weekday(day) for day in absence_df['Day']]

    absence_df.index.rename("index", inplace=True)

    pd.set_option('display.max_rows', 1000)
    print(absence_df)
    absence_df.to_csv("absence_df.csv")


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))