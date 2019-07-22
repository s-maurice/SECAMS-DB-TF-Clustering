import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time

def main():
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

    i = 0
    print(len(df['USERID'].unique()))
    for userid in df['USERID'].unique():
        user_df = df[df['USERID'] == userid]

        cur_absence_df = pd.DataFrame(columns=['USERID', 'Day', 'Present', 'Weekday'])

        cur_absence_df['Present'] = in_days_series.isin(user_df['DOY'])
        cur_absence_df['USERID'] = userid

        # absence_df = pd.concat([absence_df, cur_absence_df], sort=False)
        absence_users.append(cur_absence_df)

    absence_df = pd.concat(absence_users, sort=False)

    absence_df['Day'] = in_days * len(absence_users)
    absence_df['Weekday'] = absence_df['Day'].apply(lambda x: x.weekday() < 5)
    absence_df.reset_index(drop=True)
    print(absence_df)
    absence_df.to_csv("absence_df.csv")


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))