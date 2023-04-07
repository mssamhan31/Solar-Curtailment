#IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pytz #for timezone calculation
import math
import matplotlib.dates as md
import gc
import os
from datetime import datetime
import calendar
import seaborn as sns; sns.set()
import itertools
#import datetime
from time import gmtime, strftime
from matplotlib import cm
from IPython.display import display
#%matplotlib qt
#%matplotlib inline

#SET GLOBAL PARAMETERS
# ================== Global parameters for fonts & sizes =================
FONT_SIZE = 20
rc={'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE, 
    'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE}
plt.rcParams.update(**rc)
plt.rc('font', weight='bold')
 
# For label titles
fontdict={'fontsize': FONT_SIZE, 'fontweight' : 'bold'}
# can add in above dictionary: 'verticalalignment': 'baseline' 

style = 'ggplot' # choose a style from the above options
plt.style.use(style)


class ClearSkyDay():
    """
    A class consists of methods related to clear sky day detection

    Methods
        check_clear_sky_day : Check whether a certain date is a clear sky day based on the ghi data of that day. Needs ghi data.
        string_to_float : Remove leading and trailing space, as well as check if a variable is a null.
        get_timestamp_date_string : Convert format from YYYY-MM to YYYY_MM. The latter is used for text to input the ghi data.
        separate_ghi_data : Separate the monthly ghi data into a dict with key of date and value of ghi data.
        days_in_month : Get the number of days in a certain month
        detect_clear_sky_day : Check whether a certain day is a clear sky day or not. 
        
    """
    
    def check_clear_sky_day(self, date, file_path):
        """Check whether a certain date is a clear sky day based on the ghi data of that day. Needs ghi data.

        Args:
        date (str): date in YYYYMMDD format
        file_path (str): file_path of the ghi file

        Returns:
        clear_sky_day (bool): is it a clear sky day or not

        Funcitons needed:
        - get_timestamp_date_string
        - separate_ghi_data
        - detect_clear_sky_day

        We say that a certain day is a clear sky day
        if the ghi profile seems to be smooth. Check detect_clear_sky_day for details. It is important to note, however,
        it is possible for a site in a clear sky day to be not having a clear power profile in the D-PV data. This is
        because the location of the ghi observation station can be a bit far from the actual site. 

        IDEA: Probably it will be nice to determine whether a site is a clear sky day from the power profile and not the 
        ghi profile?
        """

        dateFile = date[:4]+'_'+ date[5:7]
        ghi = pd.read_csv(file_path +'/sl_023034_' + dateFile + ".txt")
        timestamp_date_string = self.get_timestamp_date_string(dateFile)
        separated_ghi_data = self.separate_ghi_data(timestamp_date_string, ghi)
        ghi_df = separated_ghi_data[date]
        res, average_delta_y = self.detect_clear_sky_day(ghi_df, 530)

        if res:
            #clear_sky_days.append(date)
            #overall_clear_sky_days_dict[dateFile].append(date)
            is_clear_sky_day = True
        else:
            is_clear_sky_day = False
        return is_clear_sky_day

    # REMOVE SPACES AND CHECK IF VALUE NULL
    def string_to_float(self, string):
        """Remove leading and trailing space, as well as check if a variable is a null.

        Args:
            string (str) : a variable that wants to be checked

        Returns:
            x (float) : convert to float if it's a number, and zero if it is a null. 
        """

        x = string.strip()
        if not x:
            x = 0
        else:
            x = float(x)
        return x

    # ADJUST FORMATE FOR TIMESTAMP STRINGS
    def get_timestamp_date_string(self, string):
        """Convert format from YYYY-MM to YYYY_MM. The latter is used for text to input the ghi data.

        Args:
            string (str) : year and month information in YYYY-MM format.

        Returns:
            (str) : year and month information in YYYY_MM format.         
        """

        x = string.split("_")
        return x[0] + "-" + x[1]

    # SEPARATE THE BoM GHI DATA FILES PER DAY TO SEARCH FOR CLEAR SKY DAYS
    def separate_ghi_data(self, month, ghi):
        """Separate the monthly ghi data into a dict with key of date and value of ghi data.

        Args:
            month (str) : year and month information in YYYY-MM format.
            ghi (df) : monthly ghi data without any cleaning process

        Returns:
            combined_ghi_dict (dict) : dictionary with date as the key and ghi data as the value.    

        Functions required:
        1. string_to_float
        2. days_in_month

        This function is actually no longer used anymore because we already use a day of ghi data.
        But we do not delete it just for documentation process. It does not slow the running time as well because
        it is just a function. 
        """

        ghi['ts'] = pd.to_datetime(pd.DataFrame({'year': ghi['Year Month Day Hours Minutes in YYYY'].values,
                                                        'month': ghi['MM'],
                                                        'day': ghi['DD'],
                                                        'hour': ghi['HH24'],
                                                        'minute': ghi['MI format in Local standard time']}))
        ghi.rename(columns={'Mean global irradiance (over 1 minute) in W/sq m': 'mean_ghi',
                            'Minimum 1 second global irradiance (over 1 minute) in W/sq m': 'min_ghi',
                            'Maximum 1 second global irradiance (over 1 minute) in W/sq m': 'max_ghi',
                            'Standard deviation of global irradiance (over 1 minute) in W/sq m': 'sd_ghi',
                            'Uncertainty in mean global irradiance (over 1 minute) in W/sq m': 'uncertainty_ghi'},
                   inplace=True)
        key_ghi_values = ghi[['ts', 'mean_ghi', 'min_ghi', 'max_ghi', 'sd_ghi', 'uncertainty_ghi']].copy()
        key_ghi_values['mean_ghi'] = key_ghi_values.apply(lambda row: self.string_to_float(row['mean_ghi']), axis=1)
        key_ghi_values['min_ghi'] = key_ghi_values.apply(lambda row: self.string_to_float(row['min_ghi']), axis=1)
        key_ghi_values['max_ghi'] = key_ghi_values.apply(lambda row: self.string_to_float(row['max_ghi']), axis=1)


        combined_ghi_dict = {}
        month_number = int(month.split('-')[1])

        for day in range(1, self.days_in_month(month_number) + 1):
            day_string = str(day)
            if day < 10:
                day_string = "0" + day_string

            date = month + "-" + day_string
            df = key_ghi_values.loc[key_ghi_values['ts'] > date + " 00:00:01"]
            df = df.loc[key_ghi_values['ts'] < date + " 23:59:01"]

            combined_ghi_dict[date] = df

        return combined_ghi_dict

    def days_in_month(self, month):
        """Get the number of days in a certain month

        Args:
            month (int) : month number: between 1-12

        Returns:
            (int) : number of days in a certain month
        """

        switcher = {
            1: 31,
            2: 29,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
        return switcher.get(month, 0)


    # LOOK FOR FOR SUDDEN VARIATIONS IN SOLAR INSOLATION DATA WHICH INDICATES LIKELY CLOUD COVER, AS OPPOSED TO CLEAR PARABOLIC SHAPE OF CLEAR SKY DAY GHI CURVES
    def detect_clear_sky_day(self, ghi_df, min_max_ghi):
        """Check whether a certain day is a clear sky day or not. 

        Args:
            ghi_df (df) : ghi data
            min_max_ghi (int) : the minimum value of maximum ghi. If the maximum ghi is lower than
                                this value, means there must be cloud. 

        Returns:
            (bool) : bool value if the day is clear sky day or not. 

        It will judge that it is a clear sky day if satisfying two criterias:
        1. There is no sudden change in ghi (means cloud)
        2. The maximum ghi value is higher than a certain threshold (min_max_ghi).
        """

        df_daytime = ghi_df.loc[ghi_df['mean_ghi'] > 0]

        collective_change = 0
        ghi_list = df_daytime.mean_ghi.tolist()

        for i in range(len(ghi_list)-1):
            collective_change += abs(ghi_list[i+1] - ghi_list[i])

        if len(df_daytime.index) == 0:
            return False, 0

        average_delta_y = collective_change/len(df_daytime.index)

        if average_delta_y < 5 and max(ghi_df.mean_ghi) > min_max_ghi:
            return True, average_delta_y
        else:
            return False, average_delta_y

