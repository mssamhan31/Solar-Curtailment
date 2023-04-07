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

#IMPORT FUNCTIONS 
# for package implementatoin
from solarcurtailment.energy_calculation import *
from solarcurtailment.clear_sky_day import *
from solarcurtailment.tripping_curt import *
from solarcurtailment.vvar_curt import *
from solarcurtailment.vwatt_curt import *
from solarcurtailment.polyfit import *
from solarcurtailment.file_processing import *
from solarcurtailment.data_visualization import *

#for local package testing
# from energy_calculation import *
# from clear_sky_day import *
# from tripping_curt import *
# from vvar_curt import *
# from vwatt_curt import *
# from polyfit import *
# from file_processing import *
# from data_visualization import *

#class instantiation
file_processing = FileProcessing()
clear_sky_day = ClearSkyDay()
data_visualization = DataVisualization()
energy_calculation = EnergyCalculation()
tripping_curt = TrippingCurt()
polyfit_f = Polyfit()
vvar_curt = VVarCurt()
vwatt_curt = VWattCurt()


def compute(file_path, data_file, ghi_file):
    ''' Compute solar curtailment from D-PV time series data of a certain site in a certain date & ghi data.
    
    Args:
        file_path (str) : directory path
        data_file (str) : D-PV time series data of a certain site in a certain date file name
        ghi_file (str) : ghi file name

    Returns:
        None, but displaying summary of curtailment analysis, ghi plot, power scatter plot, and power lineplot.
        
    Functions needed:
        - input_general_files
        - check_data_size
        - site_organize
        - resample_in_minute
        - check_polyfit
        - check_clear_sky_day
        - check_tripping_curtailment
        - check_energy_generated
        - check_vvar_curtailment
        - check_vwatt_curtailment
        - check_energy_expected
        - summarize_result_into_dataframe
        - display_ghi
        - display_power_scatter
        - display_power_voltage
    '''
    
    
    site_details, unique_cids= file_processing.input_general_files(file_path)
    summary_all_samples = pd.DataFrame()

    data = pd.read_csv(file_path + data_file)
    pd.to_datetime(data['Timestamp'].str.slice(0, 19, 1))
    data['Timestamp'] = pd.to_datetime(data['Timestamp'].str.slice(0, 19, 1))
    data.set_index('Timestamp', inplace=True)

    size_is_ok = file_processing.check_data_size(data)
    if not size_is_ok:
        print('Cannot analyze this sample due to incomplete data.')
    else:
        ghi = pd.read_csv(file_path + ghi_file, index_col = 0)
        ghi.index = pd.to_datetime(ghi.index)

        c_id = data['c_id'][0]
        date = str(data.index[0])[:10]

        data_site, ac_cap, dc_cap, EFF_SYSTEM, inverter = vvar_curt.site_organize(c_id, site_details, data, unique_cids)
        data_site = file_processing.resample_in_minute(data_site)

        #check the expected power using polyfit
        data_site, polyfit, is_good_polyfit_quality = polyfit_f.check_polyfit(data_site, ac_cap)
        #data_site, a, is_good_polyfit_quality = check_polyfit_constrained(data_site, ac_cap)

        is_clear_sky_day = clear_sky_day.check_clear_sky_day(date, file_path)
        tripping_response, tripping_curt_energy, estimation_method, data_site = tripping_curt.check_tripping_curtailment(is_clear_sky_day, c_id, data_site, unique_cids, ac_cap, site_details, date)    
        energy_generated, data_site = energy_calculation.check_energy_generated(data_site, date, is_clear_sky_day, tripping_curt_energy)
        vvar_response, vvar_curt_energy, data_site = vvar_curt.check_vvar_curtailment(c_id, date, data_site, ghi, ac_cap, dc_cap, EFF_SYSTEM, is_clear_sky_day)
        data_site, vwatt_response, vwatt_curt_energy = vwatt_curt.check_vwatt_curtailment(data_site, date, is_good_polyfit_quality, file_path, ac_cap, is_clear_sky_day)

        energy_generated_expected, estimation_method = energy_calculation.check_energy_expected(energy_generated, tripping_curt_energy, vvar_curt_energy, vwatt_curt_energy, is_clear_sky_day)

        summary = file_processing.summarize_result_into_dataframe(c_id, date, is_clear_sky_day, energy_generated, energy_generated_expected, estimation_method, tripping_response, tripping_curt_energy, vvar_response, vvar_curt_energy, vwatt_response, vwatt_curt_energy)

        display(summary)
        data_visualization.display_ghi(ghi, date)
        data_visualization.display_power_scatter(data_site, ac_cap)
        data_visualization.display_power_voltage(data_site, date, vwatt_response, vvar_response)