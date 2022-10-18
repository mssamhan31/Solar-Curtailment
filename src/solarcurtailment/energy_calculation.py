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

#ENERGY GENERATED CALCULATION
class EnergyCalculation():
    """
    A class consists of methods related to energy calculation

    Methods
        check_energy_generated : Get the amount of energy generated in a certain site in a certain day, unit kWh.
        check_energy_expected : Calculate the expected energy generation without curtailment and the estimation method
    """
    
    def check_energy_generated(self, data_site, date, is_clear_sky_day, tripping_curt_energy):
        """Get the amount of energy generated in a certain site in a certain day, unit kWh.

        Args:
            data_site (df): Cleaned D-PV time-series data, output of site_orgaize function
            date (str): date in focus
            is_clear_sky_day (bool): whether the date is a clear sky day or not
            tripping_curt_energy (float): the amount of energy curtailed due to tripping response

        Returns:
            energy_generated (float): Single value of the total energy generated in that day
            data_site (df): D-PV time series data with updated 'power_expected' column if the there is tripping in a non clear sky day.
        """

        #sh_idx = (data_site.index.hour>= 7) & (data_site.index.hour <= 17)
        #hour filter should not be necessary since outside of that hour, the power is zero anyway.

        date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()
        date_idx = data_site.index.date == date_dt
        energy_generated = data_site.loc[date_idx, 'power'].resample('h').mean().sum()/1000

        if not is_clear_sky_day:
            if tripping_curt_energy > 0:
                data_site['power_expected'] = data_site['power_expected_linear']

        return energy_generated, data_site


    def check_energy_expected(self, energy_generated, tripping_curt_energy, vvar_curt_energy, vwatt_curt_energy, is_clear_sky_day):
        ''' Calculate the expected energy generation without curtailment and the estimation method

        Args:
            energy_generated (float): the actual energy generated with curtailment
            tripping_curt_energy (float) : energy curtailed due to tripping. Can't be n/a
            vvar_curt_energy (float) :energy curtailed due to VVAr. Can be n/a in a non clear sky day
            vwatt_curt_energy (float) : energy curtailed due to VWatt. Can be n/a in a non clear sky day
            is_clear_sky_day (bool) : yes if the day is a clear sky day

        Returns:
            energy_generated_expected (float) : the estimated energy generated without curtailment
            estimation_method (str) : the method of estimating the previous value
        '''

        if is_clear_sky_day:
            estimation_method = 'Polyfit'
            energy_generated_expected = energy_generated + tripping_curt_energy + vvar_curt_energy + vwatt_curt_energy
        elif tripping_curt_energy > 0:
            estimation_method = 'Linear'
            if math.isnan(vvar_curt_energy):
                vvar_curt_energy = 0
            if math.isnan(vwatt_curt_energy):
                vwatt_curt_energy = 0
            energy_generated_expected = energy_generated + tripping_curt_energy + vvar_curt_energy + vwatt_curt_energy
        else:
            estimation_method = 'n/a'
            energy_generated_expected = 'n/a'

        return energy_generated_expected, estimation_method
