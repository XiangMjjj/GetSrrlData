# -*- coding: utf-8 -*-
# @Time : 2021/9/6 15:31
# @Author : Mingjun Xiang
# @Site : 
# @File : download.py
# @Software: PyCharm 
# @Illustration: The intuition of this project is that the origin file
#                only supports downloading data from limited time range
#                with limited types of data. (at least 2017/10/01 - 2021/12/31)
#                https://github.com/fengcong1992/OpenSolar_Python package/SRRL_download.py

#                The data are available from 2017/10/01.
#                (https://midcdmz.nrel.gov/apps/day.pl?BMS).
#                (https://midcdmz.nrel.gov/apps/imagergallery.pl?SRRLASI)

#                Support data includes:
#                    DATE (MM/DD/YYYY),MST,Global CMP22 (vent/cor) [W/m^2],Direct CHP1-1 [W/m^2],
#                    Diffuse CM22-2 (vent/cor) [W/m^2],Zenith Angle [degrees],Azimuth Angle [degrees],
#                    Tower Dry Bulb Temp [deg C],Data lab Dry Bulb Temp [deg C],Tower RH [%],
#                    Data lab RH [%],Total Cloud Cover [%],Opaque Cloud Cover [%],Avg Wind Speed @ 6ft [m/s],
#                    Avg Wind Speed @ 19ft [m/s],Avg Wind Speed @ 22ft [m/s],Peak Wind Speed @ 6ft [m/s]
#                    ,Peak Wind Speed @ 19ft [m/s],Peak Wind Speed @ 22ft [m/s],
#                    Avg Wind Direction @ 6ft [deg from N],Avg Wind Direction @ 19ft [deg from N],
#                    Avg Wind Direction @ 22ft [deg from N],Precipitation [mm]

#                If user want to change the types of data or limited time range,
#                look up to website at Line: 96 - 102 and rectify them, which
#                only affect ground based measurements.


from datetime import datetime, timedelta
import os
import requests
import zipfile
import pandas as pd
import io
from bisect import bisect_left


def SRRL_download(root_data, date_start, date_end, skyimg, tmseries, ifunzip, ifunique):
    if skyimg:
        lost_day = []
        lgth_day = (datetime.strptime(date_end, '%Y-%m-%d') -
                    datetime.strptime(date_start, '%Y-%m-%d')).total_seconds() / 86400 + 1
        # download sky images every day
        for no_day in range(int(lgth_day)):
            date = datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=no_day)
            date2 = date.strftime('%Y%m%d')
            year = date.strftime('%Y')
            print(date2)
            URL = r'https://midcdmz.nrel.gov/tsi/SRRLASI/' + year + '/' + str(date2) + '.zip'

            root_save = os.path.join(root_data, str(date2))
            if not os.path.exists(root_save):
                os.mkdir(root_save)

            file_name = os.path.join(root_save, str(date2) + '.zip')
            r = requests.get(URL)

            with open(file_name, 'wb') as f:
                f.write(r.content)

            open(file_name, 'wb').write(r.content)

            # whether unzip the file
            try:
                if ifunzip:
                    zip_ref = zipfile.ZipFile(file_name, 'r')
                    zip_ref.extractall(root_save)
                    zip_ref.close()
                    os.remove(file_name)
            except Exception:
                lost_day.append(date2)
                continue

            # whether delete redundant files
            if ifunique:
                list_file = os.listdir(root_save)  # files in the root
                list_file2 = pd.Series([x[0:14] for x in list_file])  # files without extensions
                list_file3 = list_file2.unique()  # unique file numbers
                # download one day and refer to that(now is normal and under exposure time without modify)
                list_file4 = pd.Series(list_file3 + '_11.jpg')
                list_file5 = pd.Series(list_file3 + '_12.jpg')
                list_file6 = list_file4.append(list_file5)
                for n_file in range(len(list_file)):
                    if not list_file[n_file] in list(list_file6):
                        os.remove(os.path.join(root_save, list_file[n_file]))
        print(lost_day, 'are lost!')

    if tmseries:
        # URL3 = 'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20171201;edy=31;emo=12;eyr=9999;year=2017;month=12;day=1;time=1;zenloc=209;inst=3;inst=55;inst=69;type=data;endyear=2018;endmonth=12;endday=31'
        # r3 = requests.get(URL3).content
        # df_2018 = pd.read_csv(io.StringIO(r3.decode('utf-8')))
        # df_2018.columns.values[[0, 2, 3, 4]] = ['Date', 'GHI', 'DNI', 'DHI']
        dfs = []
        URLs = ['https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20150101;edy=30;emo=11;eyr=2017;year=2017;month=10;day=1;time=1;zenloc=210;inst=3;inst=53;inst=69;inst=116;inst=117;inst=120;inst=123;inst=128;inst=131;inst=132;inst=133;inst=134;inst=135;inst=136;inst=139;inst=140;inst=141;inst=144;inst=145;inst=146;inst=153;type=data;endyear=2017;endmonth=11;endday=30',
                'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20171201;edy=31;emo=12;eyr=9999;year=2017;month=12;day=1;time=1;zenloc=200;inst=3;inst=55;inst=69;inst=123;inst=124;inst=127;inst=130;inst=135;inst=138;inst=139;inst=140;inst=141;inst=142;inst=143;inst=145;inst=146;inst=147;inst=149;inst=150;inst=151;inst=157;type=data;endyear=2017;endmonth=12;endday=31',
                'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20171201;edy=31;emo=12;eyr=9999;year=2018;month=01;day=1;time=1;zenloc=200;inst=3;inst=55;inst=69;inst=123;inst=124;inst=127;inst=130;inst=135;inst=138;inst=139;inst=140;inst=141;inst=142;inst=143;inst=145;inst=146;inst=147;inst=149;inst=150;inst=151;inst=157;type=data;endyear=2018;endmonth=12;endday=31',
                'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20171201;edy=31;emo=12;eyr=9999;year=2019;month=01;day=1;time=1;zenloc=200;inst=3;inst=55;inst=69;inst=123;inst=124;inst=127;inst=130;inst=135;inst=138;inst=139;inst=140;inst=141;inst=142;inst=143;inst=145;inst=146;inst=147;inst=149;inst=150;inst=151;inst=157;type=data;endyear=2019;endmonth=12;endday=31',
                'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20200101;edy=31;emo=12;eyr=9999;year=2020;month=01;day=1;time=1;zenloc=200;inst=3;inst=60;inst=74;inst=130;inst=131;inst=134;inst=137;inst=142;inst=145;inst=146;inst=147;inst=148;inst=149;inst=150;inst=152;inst=153;inst=154;inst=156;inst=157;inst=158;inst=164;type=data;endyear=2020;endmonth=12;endday=31',
                'https://midcdmz.nrel.gov/apps/plot.pl?site=BMS;start=20200101;edy=31;emo=12;eyr=9999;year=2021;month=01;day=1;time=1;zenloc=200;inst=3;inst=60;inst=74;inst=130;inst=131;inst=134;inst=137;inst=142;inst=145;inst=146;inst=147;inst=148;inst=149;inst=150;inst=152;inst=153;inst=154;inst=156;inst=157;inst=158;inst=164;type=data;endyear=2021;endmonth=12;endday=31'
                ]

        date_range = ['2017-11-30', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31']
        start_idx = bisect_left(date_range, date_start)
        end_idx = bisect_left(date_range, date_end) + 1

        for URL in URLs[start_idx:end_idx]:
            r = requests.get(URL).content
            df = pd.read_csv(io.StringIO(r.decode('utf-8')))
            df.columns.values[[0]] = ['Date']
            dfs.append(df)
        df_combine = pd.concat(dfs, axis=0)
        df_combine['Date'] = pd.to_datetime(df_combine['Date'])
        mask = (df_combine['Date'] >= datetime.strptime(str(date_start), '%Y-%m-%d')) & (
                    df_combine['Date'] <= datetime.strptime(str(date_end), '%Y-%m-%d'))
        df_final = df_combine.loc[mask]
        df_final.to_csv(os.path.join(root_data, 'SRRL_measurement_timeseries.csv'), index=False)


def SRRL_TSI_download(root_data, date_start, date_end, skyimg, ifunzip):
    if skyimg:
        lgth_day = (datetime.strptime(date_end, '%Y-%m-%d') -
                    datetime.strptime(date_start, '%Y-%m-%d')).total_seconds() / 86400
        for no_day in range(int(lgth_day)):  # int(lgth_day)
            date = datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=no_day)
            date2 = date.strftime('%Y%m%d')
            year = date.strftime('%Y')
            URL = r'https://midcdmz.nrel.gov/tsi/SRRL/' + year + '/' + str(date2) + '.zip'

            root_save = os.path.join(root_data, str(date2))
            if not os.path.exists(root_save):
                os.mkdir(root_save)
            os.chdir(root_save)

            file_name = str(date2) + '.zip'
            r = requests.get(URL)

            with open(file_name, 'wb') as f:
                f.write(r.content)

            open(file_name, 'wb').write(r.content)

            # whether unzip the file
            if ifunzip:
                zip_ref = zipfile.ZipFile(os.path.join(root_save, file_name), 'r')
                zip_ref.extractall(root_save)
                zip_ref.close()
                os.remove(file_name)


if __name__ == '__main__':
    date_start = '2017-10-01'
    date_end = '2017-10-10'

    path1 = r'./output/origin'
    if not os.path.exists(path1):
        os.makedirs(path1)
    SRRL_download(path1, date_start, date_end, skyimg=1, tmseries=1, ifunzip=True, ifunique=True)

    # path2 = r'D:\Project\pycharm\Project1\SRRL\TSI'
    # if not os.path.exists(path2):
    #     os.makedirs(path2)
    # SRRL_TSI_download(path2, date_start, date_end, skyimg=True, ifunzip=True)