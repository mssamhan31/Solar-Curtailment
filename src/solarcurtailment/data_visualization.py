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

#DATA VISUALIZATION
class DataVisualization():
    """
    A class consists of methods related to data visualization

    Methods
        display_ghi : Display GHI plot of the day
        display_power_scatter : Display P/VA rated, Q/VA rated, and PF (P/VA) as a scatter plot
        display_power_voltage : Display power, reactive power, expected power, power limit due to vwatt/vvar, and voltage
        
    """
    
    def display_ghi(self, ghi, date):
        ''' Display GHI plot of the day

        Args:
            ghi(df) : ghi data of the day
            date (str): the date of the analysis

        Returns:
            None, but displaying GHI plot
        '''

        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])

        ghi_plot = ghi[(ghi['HH24'] >= 5) & (ghi['HH24'] <= 18)]

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 5)

        ax.plot(ghi_plot['Mean global irradiance (over 1 minute) in W/sq m'], color = 'C1', markersize = 8)
        ax.set_ylabel('GHI in W/sq m', **fontdict)
        ax.set_xlabel('Time in Day', **fontdict)

        time_range = range(3,10)
        labels=[ str(2*i) + ':00' for i in time_range]
        values=[ datetime(year, month, day, 2*i, 0) for i in time_range]
        plt.xticks(values,labels)

        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)

        #plt.title('GHI Profile', **fontdict)

        plt.show()

    def display_power_scatter(self, data_site, ac_cap):
        ''' Display P/VA rated, Q/VA rated, and PF (P/VA) as a scatter plot

        Args:
            date_site(df) : time series D-PV data
            ac_cap (int) : ac capacity of the inverter

        Returns:
            None, but displaying plot
        '''

        data_site['power_normalized'] = data_site['power'] / ac_cap
        data_site['var_normalized'] = data_site['reactive_power'] / ac_cap

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 5)

        ax.scatter(data_site['voltage'], data_site['power_normalized'], color = 'r', marker = 'o', linewidths = 0.1, alpha = 1, label = 'P/VA-rated')
        ax.scatter(data_site['voltage'], data_site['var_normalized'], color = 'b', marker = 'o', linewidths = 0.1, alpha = 1, label = 'Q/VA-rated')
        ax.scatter(data_site['voltage'], data_site['pf'], color = 'g', marker = 'o', linewidths = 0.1, alpha = 1, label = 'PF')

        ax.set_xlabel('Voltage (Volt)', **fontdict)
        ax.set_ylabel('Real Power, Reactive Power, PF', **fontdict)
        ax.legend(prop={'size': 15})

        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #ax.legend(frameon=False)
        #plt.title('Real Power, Reactive Power, PF', **fontdict)

        plt.show()

    def display_power_voltage(self, data_site, date, vwatt_response, vvar_response):
        ''' Display power, reactive power, expected power, power limit due to vwatt/vvar, and voltage

        Args:
            date_site(df) : time series D-PV data
            date (str): date of analysis
            vwatt_response (str): whether there is vwatt repsonse or not
            vvar_response (str): whether there is vvar response or not

        Returns:
            None, but displaying plot
        '''

        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

        line1 = ax.plot(data_site['power'], color = 'b', label = 'Actual Power', lw = 3) 
        line2 = ax.plot(data_site['power_expected'], color = 'y', label = 'Expected Power')
        line3 = ax.plot(data_site['reactive_power'], color = 'g', label = 'Reactive Power')
        ax.set_ylim([-100, 6000])

        if vwatt_response == 'Yes':
            line4 = ax.plot(data_site['power_limit_vw'], color = 'm', label = 'Power Limit V-Watt')
            #show power limit here
        elif vvar_response == 'Yes':
            line4 = ax.plot(data_site['power_limit_vv'], color = 'm', label = 'Power Limit V-VAr')
            pass
            #show power limit here

        ax.set_ylabel('Power (watt or VAr)', **fontdict)
        ax.set_xlabel('Time in Day', **fontdict)
        ax.legend(loc = 2, prop={'size': 15})

        ax2 = ax.twinx()
        line4 = ax2.plot(data_site['voltage'], color = 'r', label = 'Voltage')
        ax2.set_ylim([199, 260])
        ax2.set_ylabel('Voltage (volt)', **fontdict)
        ax2.legend(loc = 1, prop={'size': 15})

        time_range = range(3,10)
        labels=[ str(2*i) + ':00' for i in time_range]
        values=[ datetime(year, month, day, 2*i, 0) for i in time_range]
        plt.xticks(values,labels)

        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)

        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)

        #plt.legend()
        #plt.title('Power and Voltage', **fontdict)

        plt.show()

