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

# from polyfit import Polyfit #polyfit here is a python module
# from vwatt_curt import VWattCurt

from solarcurtailment.polyfit import Polyfit
from solarcurtailment.vwatt_curt import VWattCurt

polyfit_f = Polyfit() #polyfit here is an object with class Polyfit
vwatt_curt = VWattCurt()

# from polyfit import filter_data_limited_gradients
# from polyfit import get_datetime_list
# from polyfit import get_polyfit


# from vwatt_curt import slice_end_off_df
# from vwatt_curt import filter_power_data

#VVAR CURTAILMENT PROGRAM
class VVarCurt():
    """
    A class consists of methods related to VVAr response detection and its curtailment calculation.

    Methods
        site_organize : Get a single site data and relevant meta-data information.
        check_vvar_curtailment : Check the VVAr response of a site and calculate its amount of curtailed energy.
    """
    
    def site_organize(self, c_id_idx, site_details, data, unique_cids):
        """Get a single site data and relevant meta-data information.

        Args:
            c_id_idx (int): c_id value
            site_details (df): site_details dataframe from unsw_20190701_site_details.csv file
            data (df): D-PV time-series dataframe from input_monthly_file function output
            unique_cids (df): Dataframe listing unique c_id's and their corresponding site_id

        Returns:
            data_site(df): D-PV time-series dataframe, filtered by its site_id and cleaned (polarity correction etc)
            ac_cap (float): inverter capacity in W
            dc_cap (float): PV array capacity in Wp
            EFF_SYSTEM (float): Assumed PV array efficiency between 0 and 1
            inverter (str): Concatenated string of inverter manufacturer and model        
        """

        #c_id = unique_cids.loc[c_id_idx][0]
        c_id = c_id_idx

        polarity = site_details.loc[site_details['c_id'] == c_id, 'polarity'].values[0] # get the polarity of the site
        ac_cap = site_details.loc[site_details['c_id'] == c_id, 'ac_cap_w'].values[0]
        dc_cap = site_details.loc[site_details['c_id'] == c_id, 'dc_cap_w'].values[0]
        inverter = site_details.loc[site_details['c_id'] == c_id, 'inverter_manufacturer'].values[0] + ' ' + site_details.loc[site_details['c_id'] == c_id, 'inverter_model'].values[0]

        # Extract single site data and organize: 
        data_site = data[data['c_id'] == c_id].sort_index() # get the monthly data of the specific c_id

        data_site['power'] = data_site['power'].values * polarity # polarity correction for real power
        data_site.loc[data_site['power'] < 0, 'power'] = 0 #replace negative power value into zero
        data_site['reactive_power'] = data_site['reactive_power'].values * polarity # polarity correction for reactive power

        data_site['reactive_power'] = [data_site['reactive_power'].values * -1 if np.percentile(data_site.loc[(data_site.index.hour >= 7) & (data_site.index.hour <= 17), 'reactive_power'], 75) < 0 else data_site['reactive_power'].values][0]  # double check the polarity for reactive power

        if (abs(np.percentile(data_site['reactive_power'], 99))> ac_cap) | (abs(np.percentile(data_site['reactive_power'], 1))> ac_cap): #some VAr measurements in energy format and needs to be divided by duration (i.e., 60 sec)
            # data_site['reactive_power'] =  data_site['reactive_power'].values / data_site['duration'].values # unfortunately SolA data doesn't calculate energy according to respective duration but uses a fixed 60 sec values for energy calculation
            data_site['reactive_power'] =  data_site['reactive_power'].values / 60

        data_site.index = pd.to_datetime([str(d)[0:19] for d in data_site.index]) ## convert index to make the df plottable (by removing the UTC conversion)
        data_site.sort_index(ascending = True, inplace = True) # sort the index in ascending form
        # System efficiency for calculating theoretical max output later on (use conservative loss estimates for DC power)
        EFF_INV = 0.98
        EFF_VDROP = 0.98 
        EFF_DERATING = 0.99  # module derating losses
        EFF_SYSTEM = EFF_INV * EFF_VDROP * EFF_DERATING

        # Apparent power of the inverter
        data_site['va'] = np.sqrt (data_site['power'].values**2 + data_site['reactive_power'].values**2)
        data_site['pf'] = data_site['power']/data_site['va']

        return data_site, ac_cap, dc_cap, EFF_SYSTEM, inverter

    def check_vvar_curtailment(self, c_id, date, data_site,  ghi, ac_cap, dc_cap, EFF_SYSTEM, is_clear_sky_day):
        """Check the VVAr response of a site and calculate its amount of curtailed energy. 

        Args:
            c_id (int) : circuit id value
            date (str) : date for analysis
            data_site (df): D-PV time series data sample for a certain day and site
            ghi (df): ghi data sample for a certain date
            ac_cap (int) : inverter ac capacity in watt
            dc_cap (int) : PV array capacity in wattpeak
            EFF_SYSTEM (float) : Assumed PV array efficiency between 0 and 1
            is_clear_sky_day (bool) : Bool value whether the day is a clear sky day or not based on the ghi profile

        Returns:
            vvar_response (str) : Yes, None, or Inconclusive
            vvar_curt_energy (float) : the amount of energy curtailed due to vvar response
            data_site (df) : D-PV time series data sample with added column: 'power_limit_vv', which is the maximum
                            allowed power for a given time due to the ac_cap of the inverter and the current reactive power. 

        Functions needed:
        - slice_end_of_df
        - filter_power_data
        - filter_data_limited_gradients
        - get_polyfit

        """

        date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()
        data_site_certain_date = data_site.loc[data_site.index.date == date_dt]
        ghi = ghi.loc[ghi.index.date == date_dt]
        data_site = data_site_certain_date

        # Manipulations on the original data_site to match the GHI
        dummy = data_site.copy()
        dummy.index = dummy.index.round('min')   # round the timestamp to nearest minute to match with the GHI
        dummy = dummy.groupby(level = 0 ).mean()  # average same timestamp values that fall under the same minute category

        data_site_complete = pd.DataFrame (index = ghi.index)  # create a data_site_complete with complete set of dates to match with GHI
        data_site_complete = data_site_complete.join(dummy)

        # Required conditions for V-VAr curtailment
        VAR_T = 100  # min VAr condition 
        DURATION = 60  # we have normalized all t-stamps to 60 second previously
        va_criteria = data_site_complete['va'] >= (ac_cap - VAR_T)  # this is to ensure inverter VA is close to its rated capacity (this eliminates the instances of tripping)
        var_criteria = abs(data_site_complete['reactive_power'].values) > VAR_T  # this is to ensure inverter is injecting/absorbing at least 100 vars
        curt_criteria = va_criteria & var_criteria  # curtailment criteria that satisfies the two criteria above

        data_curtailment = data_site_complete[curt_criteria]  # investigate curtailment only for the instances which satisfy above criteria 
        ghi_curtailment = ghi[curt_criteria]

        if not var_criteria.any():
            vvar_response = 'None'
        else:
            vvar_response = 'Yes'

        # max_real_power refers to what the system could generate if it wasn't curtailed
        #ISSUES FOR TROUBLESHOOTING LATER: SOMETIME MAX POWER IS LESS THAN POWER?
        if is_clear_sky_day:
            # POLYFIT METHOD TO CALCULATE THE MAX POWER WITHOUT CURTAILMENT, UNAPPLICABLE IN NON CLEAR SKY DAYS
            circuit_day_data = data_site_complete.reset_index(level=0)
            circuit_day_data.rename(columns = {'timestamp':'ts'}, inplace = True)
            circuit_day_data['ts'] = circuit_day_data['ts'].astype(str)

            df = circuit_day_data
            df = vwatt_curt.slice_end_off_df(df) # REMOVES LAST TAIL AND HEAD OF DATA AFTER IT CHANGES TO ZERO WATTS, BUT KEEPS ZERO WATT VALUES IN THE MIDDLE OF THE LIST

            df = df.loc[df['power'] > 300]

            # FILTER POWER DATA TO INCLUDE ONLY INCREASING VALUES FROM EACH SIDES (WHERE SIDES ARE DETERMINED BY EITHER SIDE OF THE MAX POWER VALUE)
            power_array, time_array = vwatt_curt.filter_power_data(df)

            # FILTER DATA SO ONLY A SUBSET OF GRADIENTS BETWEEN DATAPOINTS IS PERMITTED
            power_array, time_array = polyfit_f.filter_data_limited_gradients(power_array, time_array)

            polyfit = polyfit_f.get_polyfit(polyfit_f.get_datetime_list(time_array), power_array, 2)

            polyfit_result = pd.DataFrame({
                'timestamp' : pd.date_range(start=df['ts'].iloc[0], end=df['ts'].iloc[-1], freq='1min').astype(str)
            })
            polyfit_result['max_real_power'] = polyfit(polyfit_f.get_datetime_list(polyfit_result['timestamp']))
            polyfit_result.index = pd.to_datetime(polyfit_result['timestamp'], format='%Y-%m-%d %H:%M:%S')
            polyfit_result.drop(columns = 'timestamp', inplace = True)

            data_curtailment = pd.merge(data_curtailment, polyfit_result, left_index = True, right_index = True)
            data_curtailment ['curtailment'] = data_curtailment['max_real_power'].values - data_curtailment ['power'].values
            data_curtailment['curtailment_energy'] = data_curtailment['curtailment'].values * (DURATION/3600/1000) # Wmin to kWh energy: some sites have variable duration so finding curtailment in energy form (Wh)

            if not data_curtailment[data_curtailment['curtailment_energy'] > 0]['curtailment_energy'].sum() > 0:
                    data_curtailment['max_real_power'] = [min(ghi_t/1000 * dc_cap * EFF_SYSTEM, ac_cap) for ghi_t in ghi_curtailment['Mean global irradiance (over 1 minute) in W/sq m']]

        else: #if it is not clear sky day, use ghi to estimate maximum power without curtailmentz
            data_curtailment['max_real_power'] = [min(ghi_t/1000 * dc_cap * EFF_SYSTEM, ac_cap) for ghi_t in ghi_curtailment['Mean global irradiance (over 1 minute) in W/sq m']]
        # =============================================================================================

        data_curtailment ['curtailment'] = data_curtailment['max_real_power'].values - data_curtailment ['power'].values
        data_curtailment['curtailment_energy'] = data_curtailment['curtailment'].values * (DURATION/3600/1000) # Wmin to kWh energy: some sites have variable duration so finding curtailment in energy form (Wh)
        vvar_curt_energy = data_curtailment[data_curtailment['curtailment_energy'] > 0]['curtailment_energy'].sum()
        data_site['power_limit_vv'] = np.sqrt(ac_cap ** 2 - data_site['reactive_power']**2)
        return vvar_response, vvar_curt_energy, data_site