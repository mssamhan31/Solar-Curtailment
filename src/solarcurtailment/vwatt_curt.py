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

#VWATT CURTAILMENT PROGRAM
class VWattCurt():
    """
    A class consists of methods related to VWatt response detection and its curtailment calculation.

    Methods
        slice_end_off_df : Slice power at the beginning and at the tail of the data, where it still produce 0 power
        filter_power_data : Filter power data to include only increasing value at the first half and decreasing value at the second half. 
        volt_watt_curve : VOLT-WATT LIST BASED ON V3 INVERTER SETTING AND VOLTAGE INPUT
        check_overvoltage_avail : Check whether the maximum voltage of the data is higher than the minimum Vlimit stated in AS/NZS 4777.2
        check_energy_curtailed : Calculation of the amount of energy curtailed only in the VWatt curtailment period (expected power > max allowed power from VWatt curve)
        check_vwatt_response : Check whether the inverter shows vwatt response or not.
        check_vwatt_curtailment : Check the vwatt response and amount of curtailment due to vwatt response.
    """
    
    def slice_end_off_df(self, df):
        """Slice power at the beginning and at the tail of the data, where it still produce 0 power

        Args:
            df (df) : D-PV time series data

        Returns:
            df (df) : D-PV time series data, filtered already. 
        """

        if df is None or len(df.index) == 0:
            return None

        tmp_df = df.loc[df['power'] > 0]
        if len(tmp_df.index) == 0:
            return None

        start_time = tmp_df.iloc[0].ts
        end_time = tmp_df.iloc[len(tmp_df.index) - 1].ts

        df = df.loc[df['ts'] > start_time]
        df = df.loc[df['ts'] < end_time]

        return df

    # FILTER POWER DATA TO INCLUDE ONLY INCREASING VALUES FROM EACH SIDES (WHERE SIDES ARE DETERMINED BY EITHER SIDE OF THE MAX POWER VALUE)
    def filter_power_data(self, graph_df):
        """Filter power data to include only increasing value at the first half and decreasing value at the second half. 

        Args:
            graph_df (df) : D-PV time series data

        Returns:
            (list) : list of time value which pass the filter
            (list) : list of power value which pass the filter
        """

        if len(graph_df.index) == 0:
            return None, None

        max_daily_power = max(graph_df.power)

        if len(graph_df.loc[graph_df['power'] == max_daily_power].index) > 1:
            return None, None

        filter_array1 = []
        filter_array2 = []
        power_array = graph_df.power
        time_array = graph_df.ts

        halfFlag = True  # True is first half, False is second half
        water_mark = 0

        for curr_power in power_array:

            # IF curr_power IS GREATER THAN water_mark (LAST HIGHEST VALUE) THEN INCLUDE curr_power AND INCREASE water_mark
            if curr_power > water_mark:
                water_mark = curr_power
                filter_array1.append(True)
            else:
                filter_array1.append(False)

            if curr_power == max_daily_power:
                break

        water_mark = 0

        # PERFORM SAME FILTER ON SECOND SIDE OF POWER ARRAY
        for curr_power in power_array.iloc[::-1]:

            if curr_power == max_daily_power:
                break

            if curr_power > water_mark:
                water_mark = curr_power
                filter_array2.append(True)
            else:
                filter_array2.append(False)

        # COMBINE TO FILTERED SIDES
        filter_array2.reverse()
        filter_array = filter_array1 + filter_array2
        return power_array[filter_array], time_array[filter_array]

    def volt_watt_curve(self, v, limit):
        """VOLT-WATT LIST BASED ON V3 INVERTER SETTING AND VOLTAGE INPUT

        Args:
            v (float): voltage value
            limit (float): voltage value where the maximum allowed power starts decreasing. Could be 235-255 V.

        Returns:
            (float) : the maximum allowed cf (power/inverter capacity)
        """

        if v < limit:
            return 1
        if v < 265:
            return (1 - 0.8 * (v - limit) / (265 - limit))
        else:
            return 0


    def check_overvoltage_avail(self, data_site):
        '''Check whether the maximum voltage of the data is higher than the minimum Vlimit stated in AS/NZS 4777.2

        Args:
            data_site (df): Cleaned D-PV time-series data

        Returns:
            is_overvoltage_avail (bool) : True only if the maximum voltage of the data is higher 
                                            than the minimum Vlimit stated in AS/NZS 4777.2
        '''

        max_voltage = data_site['voltage'].max()
        min_Vlimit = 235
        if max_voltage > min_Vlimit:
            is_overvoltage_avail = True
        else:
            is_overvoltage_avail = False
        return is_overvoltage_avail

    def check_energy_curtailed(self, curtailed_data):
        """Calculation of the amount of energy curtailed only in the VWatt curtailment period (expected power > max allowed power from VWatt curve).

        Args:
            curtailed_data (df): a time series D-PV data with power and power expected columns, only in curtailment period.

        Returns:
            curt_energy (float): the curtailed energy because of VWatt, in kWh.
        """

        energy_generated_expected = vwatt_data['power_expected'].resample('h').mean().sum()/1000
        energy_generated = vwatt_data['power'].resample('h').mean().sum()/1000
        curt_energy = energy_generated_expected - energy_generated
        return curt_energy


    def check_vwatt_response(self, data_site, ac_cap):
        """Check whether the inverter shows vwatt response or not.

        This function will be done in a loop over Vlimit 235 - 255 V.
        Steps:
        1. Make a power limit value based on VW curve
        2. Filter voltage and power, which is curtailed (expected power from polyfit is higher than allowed voltage)
        3. Count the percentage of datapoints from previous step in the buffer range of VWatt curve
        4. If the percentage from the previous step is higher than certain limit, we say it shows VWatt response.

        Args:
            data_site (df) : D-PV time series data
            ac_cap(int): ac capacity of the inverter value

        Returns:
            vwatt_response (str) : Yes, None, or Inconclusive due to insufficient overvoltage datapoint.

        Functions needed:
            - volt_watt_curve

        TODO: 
        1. Reassess whether it is necessary to determine VWatt using count and gradient threshold
        2. Test for non VWatt sample & inconclusive sample
        """

        global best_percentage, best_count, best_Vlimit, vwatt_data
        #for Vlimit in list(range (246, 258)): #This is from Tim. Tim's range is different, which IDK why.
        best_percentage = 0 #initiation
        for Vlimit in list(range (235, 256)):
            #step 1. Make a power limit value based on VW curve
            data_site['power_limit_vw'] = data_site['voltage'].apply(self.volt_watt_curve, limit = Vlimit) * ac_cap

            #step 2. Filter voltage and power, which is curtailed (expected power from polyfit is higher than allowed voltage)
            global suspect_data, vwatt_data
            suspect_data_filter = data_site['power_limit_vw'] < data_site['power_expected'] 
            suspect_data = pd.DataFrame()
            suspect_data = data_site[suspect_data_filter].copy()

            #step 3. Count the percentage of datapoints from previous step in the buffer range of VWatt curve

            #create the buffer range
            BUFFER_HIGH_VAL =  150 #This is from Tim's thesis. In Tim's program the used value is 0.035 * ac_cap but IDK it doesn't work well.
            BUFFER_LOW_VAL = 150  #This is from Tim's thesis. In Tim's program the used value is 0.08 * ac_cap but IDK it doesn't work well.
            buffer_high_filter = suspect_data['power_relative'] > 0.9
            buffer_low_filter = ~buffer_high_filter

            pd.options.mode.chained_assignment = None  # default='warn'
            suspect_data.loc[buffer_high_filter, 'power_limit_upper'] = suspect_data['power_limit_vw'] + BUFFER_HIGH_VAL
            suspect_data.loc[buffer_high_filter, 'power_limit_lower'] = suspect_data['power_limit_vw'] - BUFFER_HIGH_VAL

            suspect_data.loc[buffer_low_filter, 'power_limit_upper'] = suspect_data['power_limit_vw'] + BUFFER_LOW_VAL
            suspect_data.loc[buffer_low_filter, 'power_limit_lower'] = suspect_data['power_limit_vw'] - BUFFER_LOW_VAL

            #count points in buffer
            is_low_ok = suspect_data['power_limit_lower'] < suspect_data['power']
            is_upp_ok = suspect_data['power'] < suspect_data['power_limit_upper']
            suspect_data['is_in_buffer_range'] = is_low_ok & is_upp_ok
            count_in_buffer_range = suspect_data['is_in_buffer_range'].values.sum() #count true in a col
            try:
                percentage_in_buffer_range = float(count_in_buffer_range) / float(len(suspect_data.index)) * 100
                #put the best VWLimit stats
                if percentage_in_buffer_range > best_percentage or best_percentage == 0:
                    best_percentage = percentage_in_buffer_range
                    best_count = count_in_buffer_range
                    best_Vlimit = Vlimit
                    vwatt_data = suspect_data
            except:
                pass

        data_site['power_limit_vw'] = data_site['voltage'].apply(self.volt_watt_curve, limit = best_Vlimit) * ac_cap

        #step 4. If the percentage from the previous step is higher than certain limit, we say it shows VWatt response.
        PERCENTAGE_THRESHOLD = 84
        COUNT_THRESHOLD = 30 #Tim uses 150 for a month data, where a month usually consist of around 5 clear sky days.
        #print(best_percentage)
        #print (best_Vlimit)
        if (best_percentage > PERCENTAGE_THRESHOLD) & (best_count > COUNT_THRESHOLD): #Tim uses count threshold and gradient threshold. I am not sure whether it is necessary.
            vwatt_response = 'Yes'
            vwatt_curt_energy = self.check_energy_curtailed(vwatt_data)
        elif suspect_data['voltage'].max() < 255:
            vwatt_response = 'Inconclusive due to insufficient data points'
            vwatt_curt_energy = float('nan')
        else: #no Vlimit results a good fit in all possible Vlimit value
            vwatt_response = 'None'
            vwatt_curt_energy = 0

        return vwatt_response, vwatt_curt_energy

    def check_vwatt_curtailment(self, data_site, date, is_good_polyfit_quality, file_path, ac_cap, is_clear_sky_day):
        """Check the vwatt response and amount of curtailment due to vwatt response. 

        Args:
            data_site (df) : D-PV time series data
            date (str) : date
            is_good_polyfit_quality (bool) : whether the certain date is a clear sky day or not
            file_path (str): file path where the data is saved
            ac_cap(int): ac capacity of the inverter value
            is_clear_sky_day(bool): whether it is a clear sky day or not

        Returns:
            data_site (df) : D-PV time series data, probably better to be removed before because redundant
            vwatt_response (str) : Yes, None, or Inconclusive due to insufficient overvoltage datapoint.
            vwatt_curt_energy (float) : The amount of energy curtailed due to V-Watt response. 

        Functions needed:
            - check_overvoltage_avail
            - check_vwatt_response
        """

        #check if clear sky day. This contains redundant steps like making ghi dict for all days etc, can still be improved.
        #is_clear_sky_day = check_clear_sky_day(date, file_path) 

        global vwatt_data
        vwatt_data = pd.DataFrame() #this is redundant, probably remove later.

        if not is_clear_sky_day:
            vwatt_response = 'Inconclusive due to non clear sky day.'
            vwatt_curt_energy = float('nan')
            #print('Not clear sky day')
            return data_site, vwatt_response, vwatt_curt_energy

        if not is_good_polyfit_quality:
            vwatt_response = 'Inconclusive due to poor power data'
            vwatt_curt_energy = float('nan')
            print('Polyfit quality is not good enough')
            return data_site, vwatt_response, vwatt_curt_energy

        #check overvoltage sufficiency
        is_overvoltage_avail = self.check_overvoltage_avail(data_site)

        if not is_overvoltage_avail:
            vwatt_response = 'Inconclusive due to insufficient overvoltage datapoint.'
            vwatt_curt_energy = float('nan')
            print('No voltage point over 235 V')
            return data_site, vwatt_response, vwatt_curt_energy

        #check vwatt-response here
        vwatt_response, vwatt_curt_energy = self.check_vwatt_response(data_site, ac_cap)

        return data_site, vwatt_response, vwatt_curt_energy