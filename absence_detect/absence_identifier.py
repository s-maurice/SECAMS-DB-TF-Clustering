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

    # Get rid of inconsistent data points
    start_date = dt.date(year=2016, month=5, day=1)
    end_date = dt.date(year=2016, month=7, day=1)

    df = df[df['DOY'] > start_date]
    df = df[df['DOY'] < end_date]

    in_days = []
    for date in (start_date + dt.timedelta(days=n) for n in range((end_date - start_date).days)):
        in_days.append(date)

    in_days_series = pd.Series(in_days)
    absence_users = []

    for userid in df['USERID'].unique():
        user_df = df[df['USERID'] == userid]

        cur_absence_df = pd.DataFrame(columns=['USERID', 'Day', 'Present', 'Weekday'])

        cur_absence_df['Present'] = in_days_series.isin(user_df['DOY'])
        cur_absence_df['USERID'] = userid

        absence_users.append(cur_absence_df)

    absence_df = pd.concat(absence_users, sort=False)

    absence_df['Day'] = in_days * len(absence_users)
    absence_df['Weekday'] = absence_df['Day'].apply(lambda x: is_weekday(x))
    # absence_df.reset_index(drop=True, inplace=True)
    print(absence_df)
    absence_df.to_csv("absence_df.csv")


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))