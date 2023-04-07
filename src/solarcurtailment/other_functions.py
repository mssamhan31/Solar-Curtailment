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


# This file consists of some functions which are not used, but we keep it just in case

#from EnergyCalculation
def check_energy_expected_generated(data_site, date):
    """Get the amount of expected energy generated in a certain site in a certain day, unit kWh.

    Args:
        data_site (df): Cleaned D-PV time-series data, with power_expected column
        date (str): date in focus

    Returns:
        energy_generated_expected (float): Single value of the total expected energy generated in that day
    """
    
    #sh_idx = (data_site.index.hour>= 7) & (data_site.index.hour <= 17)
    #hour filter should not be necessary since outside of that hour, the power is zero anyway.
    
    date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()
    date_idx = data_site.index.date == date_dt
    energy_generated_expected = data_site.loc[date_idx, 'power_expected'].resample('h').mean().sum()/1000
    return energy_generated_expected

#from FileProcessing 
def input_monthly_files(file_path, data_date_idx):
    """Open time-series D-PV data and ghi data of a certain month. Only compatible for SoLA data format.

    Args:
        file_path (str): The file location of the data
        data_date_idx (str): The month of the files in format 'YYYYMM' eg '201907'

    Returns:
        data (df): the opened & cleaned time-series D-PV data
        ghi (df): the opened & cleaned ghi data
        data_ori (df): the opened & unmodified time-series D-PV data
        ghi_ori (df): the opened & unmodified ghi data
    """
    
    data_path = file_path + r"/processed_unsw_" + data_date_idx + '_data_raw.csv'
    data_ori = pd.read_csv(data_path)
    data = data_ori.set_index('utc_tstamp')

    # Convert timestamp to local Adelaide time
    data.index = pd.to_datetime(data.index) # convert index from object type to datetime
    Adelaide_local_time = pytz.timezone('Australia/Adelaide')
    data.index = data.index.tz_localize(pytz.utc).tz_convert(Adelaide_local_time) # convert utc to local adelaide time
    data.index.rename('Timestamp', inplace = True)

    # Load GHI data
    ghi_date_idx = data_date_idx[0:4] + '_' + data_date_idx[4:]
    ghi_path = file_path + r"/sl_023034_" + ghi_date_idx +'.txt'
    ghi = pd.read_csv (ghi_path) 
    ghi_ori = ghi.copy()

    ghi['timestamp'] = pd.to_datetime(pd.DataFrame ({'year' : ghi['Year Month Day Hours Minutes in YYYY'].values, 
                                                    'month' : ghi['MM'], 
                                                    'day' : ghi['DD'], 
                                                   'hour' : ghi['HH24'], 
                                                   'minute' : ghi['MI format in Local standard time']}))
    ghi.set_index('timestamp', inplace = True)
    # Deal with the space characters (ghi is in object/string form at the moment)
    ghi['Mean global irradiance (over 1 minute) in W/sq m'] = [float(ghi_t) if ghi_t.count(' ')<= 3 else np.nan for ghi_t in ghi['Mean global irradiance (over 1 minute) in W/sq m']]
    return data, ghi, data_ori, ghi_ori

def filter_date(data, date):
    """Filter any df with timestamp as an index into df in a certain date.

    Args:
        data (df): df with timestamp as an index
        date (str): date

    Returns:
        data (df): filtered data
    """
    
    date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()
    data = data[data.index.date == date_dt] #focus only on the date
    return data

def isfloat(num):
    ''' Check whether a variable is a number or not
    
    Args:
        num : variable that is wanted to be checked

    Returns:
        (bool): True if a number, False if not a number
    '''
    
    try:
        float(num)
        return True
    except ValueError:
        return False
    
#from Polyfit
def func(a,x):
    """Calculate the result of a quadratic function

    Args:
    a (nd array of dimension 3x1) : a[0] is coefficient of x^2, a[1] is coefficient of x, a[2] is the constant
    x (nd array of dimension nx1) : matrix of x value that will be plugged into the function, n is the number of x values

    Returns:
    y (nd array of dimension nx1) : matrix of result value, n is the number of y values
    """
    y = a[0] * x**2 + a[1] * x + a[2]
    return y

def sum_squared_error(a):
    """Calculate the sum of the square error of the fitting result and the actual value

    Args:
    a (nd array of dimension 3x1) : a[0] is coefficient of x^2, a[1] is coefficient of x, a[2] is the constant
    
    Returns:
    sum_squared_error (float) : a single value of sum squared error. This will be used for the objective value that we
                                want to minimize for the fitting process.
    """
    
    y_fit = func(a,x_for_fitting) #x_for fitting here is a global variable so must be defined before declaring the function.
    sum_squared_error = sum((y_fit - y)**2)
    return sum_squared_error

def check_polyfit_constrained(data, ac_cap):
    """Get the expected generated power, with constrain must be at least the same with the actual power.

    Args:
    data (df) : D-PV time series data with power data
    ac_cap (int): The maximum real power generated by the pv system due to inverter limitation
    
    Returns:
    data (df) : D-PV time series data, filtered sunrise sunset, added with 'power_expected' column and 'power_relative' column
    a (list) : polyfit result in terms of the coefficient of x^2, x, and the constant
    is_good_polyfit_quality (bool) : whether the polyfit quality is good enough or not.
    
    functions needed:
    - filter_sunrise_sunset
    - filter_curtailment
    - func
    - sum_squared_error
    
    """
    
    from scipy.optimize import minimize
    from scipy.optimize import NonlinearConstraint
    from scipy.optimize import fmin
    import warnings
    warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
        
    data_site['unix_ts'] = data_site.index.astype(int) / 10**9
    data_site['x_fit'] = (data_site['unix_ts'] - data_site['unix_ts'][0])/60 #this is necessary for the fitting purpose since the precision is broken if we use alarge number.

    data['power_relative'] = data['power'] / ac_cap
    VA_W_RATIO = 1.125
    data['power_limit_vv'] = np.sqrt((VA_W_RATIO*ac_cap)**2 - data['reactive_power']**2)
    sunrise, sunset, data = filter_sunrise_sunset(data)
    
    global POWER_LIMIT_FITTING
    #POWER_LIMIT_FITTING = 3500
    #POWER_LIMIT_FITTING = 500
    #POWER_LIMIT_FITTING = 300
    POWER_LIMIT_FITTING = 1/2*data['power'].max()
    data_for_fitting = data.loc[data['power'] > POWER_LIMIT_FITTING] 
    #this improves the polyfit quality because in the morning the gradient is still increasing, while quadratic model has only
    #decreasing gradient.
    
    global y
    x, y = filter_curtailment(data_for_fitting)
    
    global x_for_fitting
    x_for_fitting = np.array(x)
    y_for_fitting = np.array(y)

    #Set the constraint: the polyfit result - actual power >= 0
    con_func_1 = lambda x: func(a = x, x = x_for_fitting) - y_for_fitting
    lower_bound = NonlinearConstraint(con_func_1, 0, np.inf)

    #Perform the fitting using scipy.optimize.minimize, 'trust-constr' is chosen because we have constrain to add
    res = minimize(sum_squared_error, x0 = [0, 0, 0], method = 'trust-constr', constraints = lower_bound)
    a = res.x #this is the fitting result (quadratic function coefficient)

    data['power_expected'] = func(a, np.array(data['x_fit']))
    
    error = abs(data['power_expected'] - data['power'])
    points_near_polyfit_count = error[error<100].count()

    if points_near_polyfit_count > 50: #the initial value is 50
        is_good_polyfit_quality = True
    else:
        is_good_polyfit_quality = False
    
    #this is for adjusting the power expected in the morning and evening where P < 1000
    data.loc[data['power_expected'] < data['power'], 'power_expected'] = data['power']
    
    #limit the maximum power expected to be the same with ac capacity of the inverter
    data.loc[data['power_expected'] > ac_cap, 'power_expected'] = ac_cap
        
    return data, a, is_good_polyfit_quality 

#from TrippingCurt
def get_penetration_by_postcode(PC_INSTALLS_DATA_FILE_PATH, DWELLINGS_DATA_FILE_PATH, sum_stats_df, output_df):
    """Get the number of pv penetration based on its postcode.

    Args:
    PC_INSTALLS_DATA_FILE_PATH (str): the file path where there is relevant data files
    DWELLINGS_DATA_FILE_PATH (str): the file path where there is relevant data files
    sum_stats_df (df): summary stats
    output_df (df): main output df

    Returns:
    sum_stats_df (df): summary stats with penetration info added as a column
    """
    
    cer_data = pd.read_csv(PC_INSTALLS_DATA_FILE_PATH, index_col = 'Small Unit Installation Postcode')
    apvi_data = pd.read_csv(DWELLINGS_DATA_FILE_PATH)
    # Need to calculate the cumulative num of installs in each month first
    # First clean CER data
    cer_data['Previous Years (2001- June 2019) - Installations Quantity'] = cer_data['Previous Years (2001- June 2019) - Installations Quantity'].str.replace(',', '')
    cer_data['Previous Years (2001- June 2019) - Installations Quantity'] = cer_data['Previous Years (2001- June 2019) - Installations Quantity'].astype(str).astype(int)
    # Then get cumulative
    cer_data['Jul'] = cer_data['Previous Years (2001- June 2019) - Installations Quantity'].astype(int) + cer_data['Jul 2019 - Installations Quantity'].astype(int)
    cer_data['Aug'] = cer_data['Jul'] + cer_data['Aug 2019 - Installations Quantity']
    cer_data['Sep'] = cer_data['Aug'] + cer_data['Sep 2019 - Installations Quantity']
    cer_data['Oct'] = cer_data['Sep'] + cer_data['Oct 2019 - Installations Quantity']
    cer_data['Nov'] = cer_data['Oct'] + cer_data['Nov 2019 - Installations Quantity']
    cer_data['Dec'] = cer_data['Nov'] + cer_data['Dec 2019 - Installations Quantity']
    cer_data['Jan'] = cer_data['Dec'] + cer_data['Jan 2020 - Installations Quantity']
    cer_data['Feb'] = cer_data['Jan'] + cer_data['Feb 2020 - Installations Quantity']
    cer_data['Mar'] = cer_data['Feb'] + cer_data['Mar 2020 - Installations Quantity']
    cer_data['Apr'] = cer_data['Mar'] + cer_data['Apr 2020 - Installations Quantity']

    # get month in this data set
    month_now = output_df.index[0]
    month_now = month_now.strftime("%b")

    # Extract just the month in this data set
    num_installs = cer_data[[str(month_now)]]
    # Rename from 'Jan' etc. to num installs
    num_installs = num_installs.rename(columns = {str(month_now) : str(month_now) + '_cumulative_num_installs'})
    num_dwellings = apvi_data[['postcode','estimated_dwellings']]

    # Merge num installs and num dwellings onto sum_stats_df and return this df
    sum_stats_df = sum_stats_df.reset_index().merge(num_installs, left_on='s_postcode', right_index=True, how='left').set_index('site_id')
    sum_stats_df = sum_stats_df.reset_index().merge(num_dwellings, left_on='s_postcode', right_on='postcode', how='left').set_index('site_id')
    sum_stats_df = sum_stats_df.drop(['postcode'], axis=1)
    sum_stats_df['pv_penetration'] = sum_stats_df[str(month_now) + '_cumulative_num_installs'] / sum_stats_df['estimated_dwellings']
    return sum_stats_df


def distribution_plot_1(all_days_df, data_date_list, ax, colour_list):
    """Show a distribution plot the sites with curtailment for some dates

    Args:
    all_days_df (df): complete data for all days
    data_date_list (df): date list for analysis, Naomi seperates the non clear sky days and clear sky days
    ax (ax) : the axis for plotting
    colour_list (list): list of color used for plotting

    Returns:
    -
    
    Side effect:
    distribution plot
    
    """
    # Used to look like this: ax.plot(a['proportion_of_sites'], a['percentage_lost'], 'o-', markersize=4, linewidth=1, label=data_date_list[0], c=colour_list[0])
    # Have generalised so as to pass it all_days_df instead of 'a', 'b' etc.
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[0]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[0]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label=data_date_list[0], c=colour_list[0])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[1]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[1]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label=data_date_list[1], c=colour_list[0])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[2]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[2]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[2], c=colour_list[1])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[3]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[3]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[3], c=colour_list[1])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[4]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[4]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[4], c=colour_list[2])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[5]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[5]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[5], c=colour_list[2])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[6]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[6]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[6], c=colour_list[3])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[7]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[7]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[7], c=colour_list[3])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[8]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[8]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[8], c=colour_list[4])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[9]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[9]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[9], c=colour_list[4])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[10]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[10]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[10], c=colour_list[5])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[11]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[11]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[11], c=colour_list[5])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[12]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[12]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[12], c=colour_list[6])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[13]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[13]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[13], c=colour_list[6])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[14]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[14]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[14], c=colour_list[7])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[15]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[15]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[15], c=colour_list[7])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[16]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[16]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[16], c=colour_list[8])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[17]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[17]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[17], c=colour_list[8])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[18]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[18]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[18], c=colour_list[9])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[19]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[19]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[19], c=colour_list[9])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[20]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[20]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[20], c=colour_list[10])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[21]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[21]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[21], c=colour_list[10])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[22]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[22]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[22], c=colour_list[11])
    ax.plot(all_days_df[all_days_df['data_date'] == data_date_list[23]]['proportion_of_sites'], all_days_df[all_days_df['data_date'] == data_date_list[23]]['percentage_lost'], 'o-', markersize=3, linewidth=0.5, label= data_date_list[23], c=colour_list[11])

    # # get average by data date and plot on top in black
    # all_days_df_average = pd.DataFrame(all_days_df.groupby('site_id')['percentage_lost'].mean(),columns=['percentage_lost'])
    # all_days_df_average = all_days_df_average.sort_values('percentage_lost', ascending =False)
    # # Get % of systems
    # all_days_df_average['proportion_of_sites'] = range(len(all_days_df_average))
    # all_days_df_average['proportion_of_sites'] = (all_days_df_average['proportion_of_sites'] + 1) / len(all_days_df_average)
    # print(all_days_df_average)
    # # Add to plot
    # ax.plot(all_days_df_average['proportion_of_sites'], all_days_df_average['percentage_lost'], 'o-', markersize=4, linewidth=1, label= 'Average percentage lost by site', c='black')

    # Show worst case line
    rect = Rectangle((-0.05, -0.05), 0.1, 0.21, linewidth=1, edgecolor='black', facecolor='none', linestyle='--',zorder=25)
    ax.add_patch(rect)
    # get legend
    legend = ax.legend()
    plt.legend(ncol=3, title="Date", prop={'size': 12})
    # set y axis to percentage
    # vals = ax.get_yticks()
    # ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # set x axis as percentage
    # vals = ax.get_xticks()
    # ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    # Axis labels
    plt.xlabel('Proportion of sites (all sites)')
    plt.ylabel('Estimated generation curtailed')
    # Axis limits
    # ax.set(ylim=(-0.0001, 0.7))
    
#from VWattCurt
def get_telemetry_string(string):
    """Convert month and year data into format that is the same with ghi filename. 

    Args:
    string (str): month and year in YYYYMM format.

    Returns:
    (str) : month and year in YYYY_MM format, used for text to call ghi data. 
    """
    
    x = string.split("_")
    return x[0] + x[1]

def filter_data_clear_sky_days(data, clear_sky_days):
    """Filter data to include only the clear sky days. 

    Args:
    data (df): D-PV time series data
    clear_sky_days (series): list of clear sky days in the certain month. 

    Returns:
    filtered_df (df) : the filtered D-PV time series data, to include only clear sky days. 
    
    This is initially written for Tim's script, which anlayzes the data in monthly basis for the 
    vwatt response detection. However, this is not used anymore because in this script we analyze the 
    data in daily basis. 
    """
    
    filtered_df = None
    
    for day in clear_sky_days:
        tmp_df = data.loc[data['utc_tstamp'] > convert_sa_time_to_utc(day + " 00:00:01")]
        tmp_df = tmp_df.loc[tmp_df['utc_tstamp'] < convert_sa_time_to_utc(day + " 23:59:01")]

        if filtered_df is None:
            filtered_df = tmp_df
        else:
            filtered_df = filtered_df.append(tmp_df, ignore_index=True)

    return filtered_df

def convert_sa_time_to_utc(sa_time):
    """convert time from South Australia zone to UTC zone.

    Args:
    sa_time (str): time in South Australia zone

    Returns:
    a (str) : time in UTC
    
    UTC is GMT 0, while SA is GMT+9:30.
    """
    
    timeFormat = "%Y-%m-%d %H:%M:%S"
    x = datetime.strptime(sa_time, timeFormat)
    sa_local_time = pytz.timezone('Australia/Adelaide')
    utc_time = pytz.utc
    sa_moment = sa_local_time.localize(x, is_dst=None)
    utc_time = sa_moment.astimezone(utc_time)
    a = utc_time.strftime(timeFormat)
    return a

def organise_sites(clear_sky_days, site_id_list, month, inverter_telemetry, site_details, cicuit_details):  # add gen data
    """organises all telemetry in the hierarchy: site->circuits->days_of_data

    Args:
    clear_sky_days (list) : List of clear sky days of a certain month
    site_id_list (list) : List of site id with overvoltage datapoints
    month (str) : month
    inverter_telemetry (str) : Month in YYYY_MM format
    site_details (df) : site_details data
    cicuit_details (df) : circuit details data

    Returns:
    overall_site_organiser (dict) : Dict of all sites which have overvoltage datapoints. Keys are site_id,
                                    values are site (a variable with object Site)
    
    Functions required:
    organise_individual_site
    """
    
    overall_site_organiser = {}

    for site_id in site_id_list:
        if site_id not in site_details.site_id.unique():
            continue
        overall_site_organiser[site_id] = organise_individual_site(clear_sky_days, site_id, month, inverter_telemetry,
                                        site_details.loc[site_details['site_id'] == site_id],
                                        cicuit_details.loc[
                                            cicuit_details['site_id'] == site_id])

    return overall_site_organiser


def organise_individual_site(clear_sky_days, site_id, month, inverter_telemetry, site_details, cicuit_details):
    """Organise c_id data into site class

    Args:
    clear_sky_days (list) : List of clear sky days of a certain month
    site_id (int) : site_id
    month (str) : month in YYYY-MM format
    inverter_telemetry (df) : D-PV time-series data
    site_details (df) : site_details data
    cicuit_details (df) : circuit details data

    Returns:
    site (Site class)
    
    Functions/Class Required:
    organise_individual_circuit
    """
    
    site = Site(site_id, site_details.iloc[0].s_postcode, site_details.iloc[0].pv_install_date,
                site_details.iloc[0].ac_cap_w,
                site_details.iloc[0].dc_cap_w, site_details.iloc[0].inverter_manufacturer,
                site_details.iloc[0].inverter_model) #initiating an object of class Site

    for row in cicuit_details.iterrows():
        c_id = row[1].c_id #assigning the c_id
        site.c_id_data[c_id] = organise_individual_circuit(clear_sky_days, c_id, site_id, month,
                                inverter_telemetry.loc[inverter_telemetry['c_id'] == c_id],
                                row[1].con_type, row[1].polarity) #store the circuit data into variable site

    return site


def organise_individual_circuit(clear_sky_days, c_id, site_id, month, inverter_telemetry, con_type, polarity):
    """organise circuit data into circuit class

    Args:
    clear_sky_days (list) : List of clear sky days of a certain month
    c_id (int) : c_id
    site_id (int) : site_id
    month (str) : month in YYYY-MM format
    inverter_telemetry (df) : D-PV time-series data
    con_type (str): Not sure about this. Seems like one of the column in circuit data
    polarity (str): The polarity of the telemetry sensor, 1 or -1. Actual power is measured power x polarity.

    Returns:
    circuit (Circuit class): contains a D-PV data for a certain day for a certain circuit
    
    Functions/Class Required:
    Circuit
    organise_individual_day
    """
    
    circuit = Circuit(c_id, site_id, con_type, polarity) #defining a variable circuit, having a class Circuit
    inverter_telemetry['ts'] = inverter_telemetry.apply(lambda row: convert_to_sa_time(row['utc_tstamp']), axis=1)


    month_number = int(month.split('-')[1]) #not really sure what is this for..
    for day in clear_sky_days:
        #create a D-PV time-series data for a certain date for a certain site
        circuit.day_data[day] = organise_individual_day(day, inverter_telemetry) 

    return circuit


def organise_individual_day(date, inverter_telemetry):
    """filter D-PV data for only a certain date.

    Args:
    date(str) : date
    inverter_telemetry (df): D-PV time series data

    Returns:
    (df) : D-PV data filtered for a certain date, and sorted by its timestmap.
    """
    
    inverter_telemetry = inverter_telemetry.loc[inverter_telemetry['ts'] > date + " 00:00:01"]
    inverter_telemetry = inverter_telemetry.loc[inverter_telemetry['ts'] < date + " 23:59:01"]    
    return inverter_telemetry.sort_values('ts', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='first', ignore_index=False, key=None)

def find_over_voltage_sites(v, clear_sky_data, cicuit_details):
    """Determine subsets of sites that experience over voltages to different extents for later selection

    Args:
        v (float): voltage value, but isn't what for. Seems useless.
        clear_sky_data (df): D-PV time series data for clear sky day
        cicuit_details (df): circuit details, consisting c_id and site_id.

    Returns:
        site_id_list_ov (dict) : dict, key is voltage limit, and value is list of overvoltage site_id
    """
    
    site_id_list_ov = {}
    
    test_vs = list(range(235, 256))
    for i in test_vs:
        site_id_list_ov[i] = []

    c_id_list = clear_sky_data.c_id.unique()

    for c_id in c_id_list:
        c_id_to_site_id(c_id, cicuit_details)

    for c_id in c_id_list:

        df = clear_sky_data.loc[clear_sky_data['c_id'] == c_id]
        if len(df.index) == 0:
            continue

        df = df.loc[df['power'] > 0]
        if len(df.index) == 0:
            continue

        maxV = max(df.voltage)

        site_id = c_id_to_site_id(c_id, cicuit_details)

        for i in test_vs:
            if maxV > i:
                if c_id not in site_id_list_ov[i]:
                    site_id_list_ov[i].append(site_id)

    for i in test_vs:
        print("Length vMax > " + str(i) + ": " + str(len(site_id_list_ov[i])))

    return site_id_list_ov

# REUTRN THE SITE ID THAT CORRESPONDS TO A GIVEN CIRCUIT ID
def c_id_to_site_id(c_id, cicuit_details):
    """Check the site id of a certain circuit id. 

    Args:
        c_id (int): circuit id number
        circuit_details (df): circuit details data, which includes c_id and site_id column. 

    Returns:
        (int) : site id number.
    """
    
    return cicuit_details.loc[cicuit_details['c_id'] == c_id].iloc[0].site_id

# CONVERT TIMESTAMP STRINGS FROM UTC TO LOCAL SOUTH AUSTRALIA TIME, TODO: ADJUST FOR TELEMETRY ANALYSIS IN OTHER CITIES
def convert_to_sa_time(utc_tstamp):
    """convert time from UTC zone to South Australia zone.

    Args:
    utc_tstamp (str): time in UTC

    Returns:
    a (str) : time in SA zone
    
    UTC is GMT 0, while SA is GMT+9:30.
    """
    
    TIME_FORMAT_1 = "%Y-%m-%d %H:%M:%S.%f"
    TIME_FORMAT_2 = "%Y-%m-%d %H:%M:%S"
    x = datetime.strptime(utc_tstamp, TIME_FORMAT_1)
    adelaide_local_time = pytz.timezone('Australia/Adelaide')
    utc_time = pytz.utc
    utc_moment = utc_time.localize(x, is_dst=None)
    adelaide_local_time = utc_moment.astimezone(adelaide_local_time)
    a = adelaide_local_time.strftime(TIME_FORMAT_2)
    return a

def assess_volt_watt_behaviour_site(site, clear_sky_days, overall_volt_watt_dict):
    """ASSESS AGGREGATED V-WATT DATA FOR A SITE

    Args:
        site (Site) : a Site object
        clear_sky_days (list) : : list of clear sky days
        overall_volt_watt_dict (dict) : keys are the c_id, values are dict -> keys are v, p, d, values are the values.

    Returns:
        None, but it will modify site object by appending circuit data into it.
        
    Functions needed:
    assess_volt_watt_behaviour_circuit
    """
    
    for c_id in site.c_id_data.keys():
        circuit = site.c_id_data[c_id]
        assess_volt_watt_behaviour_circuit(circuit, clear_sky_days, site.dc_cap_w, site.ac_cap_w, overall_volt_watt_dict)

def assess_volt_watt_behaviour_circuit(circuit, clear_sky_days, dc_cap_w, ac_cap_w, overall_volt_watt_dict):
    """Organize the filtered V and P data in a dictionary, used for all dates in the clear sky days.

    Args:
        circuit (Circuit) : a Circuit object
        clear_sky_days (list) : : list of clear sky days
        dc_cap_w (float) : tbh I thought this is the same with ac_cap_w, but seems different. Maybe it's max PV power
        ac_cap_w (float) : inverter power capacity (watt)
        overall_volt_watt_dict (dict) : keys are the c_id, values are dict -> keys are v, p, d, values are the values.

    Returns:
        None, but it will modify overall_volt_watt_dict by appending its values. v and p are points in the VWatt curve buffer.
        
    Functions needed:
    - append_volt_watt_behaviour_data
    - display_day
    """
    
    for date in clear_sky_days:
        volt_array, relative_watt_array, filtered_time_array, filtered_power_array = append_volt_watt_behaviour_data(circuit.day_data[date], circuit.c_id, date, ac_cap_w)
        if volt_array is not None:
            
            display_day(circuit.c_id, date, circuit.day_data[date], ac_cap_w, volt_array, relative_watt_array, filtered_time_array, filtered_power_array)
            if circuit.c_id not in overall_volt_watt_dict.keys():
                overall_volt_watt_dict[circuit.c_id] = {"v": [], 'p': [], 'd': 0}

            overall_volt_watt_dict[circuit.c_id]['v'] += volt_array
            overall_volt_watt_dict[circuit.c_id]['p'] += relative_watt_array
            overall_volt_watt_dict[circuit.c_id]['d'] += 1
    print("Length of sites determined to be assessable: " + str(len(overall_volt_watt_dict.keys())))

def append_volt_watt_behaviour_data(df, c_id, date, dc_cap_w):    
    """ORGANISE DATA FOR DETERMINING COMPLIANCE FUNCTION BELOW

    Args:
        df (df) : D-PV time series data
        c_id (int) : c_id
        date (str) : date
        dc_cap_w (float) : inverter power capacity in watt

    Returns:
        volt_array_compliance (list) : list of volt_array in the buffer range
        relative_watt_array_compliance (list) : list of relative_watt_array in the buffer range
        filtered_time_array (list) : list of time filtered by removing outliers
        filtered_power_array (list) : list of power filtered by removing outliers
        
    Functions needed:
    - slice_end_off_df
    - filter_power_data
    - filter_data_limited_gradients
    - get_polyfit
    - filter_array
    - change_w_to_kw
    - determine_compliance
    - get_max_volt_watt_curve
    """
    
    if df is None:
        return None, None, None, None

    if len(df.index) == 0:
        return None, None, None, None
    
    if max(df.power) < 0.3:
        return None, None, None, None

    df = slice_end_off_df(df)

    df = df.loc[df['power'] > 300]

    if len(df.index) < 20:
        return None, None, None, None

    # Filter power data for only 'uncurtailed instances' (estimation as it is unknown when inverter is actively curtailing output)
    power_array, time_array = filter_power_data(df)

    # Filter data for limited gradients, useful in creating more accurate polyfits, as determined by visual verification
    power_array, time_array = filter_data_limited_gradients(power_array, time_array)

    if power_array is None or len(power_array) < 20:
        return None, None, None, None

    # Get polyfit estimation
    polyfit = get_polyfit(get_datetime_list(time_array), power_array, 2)

    # Simple filter for very high and low values to aid in displaying data in figures
    filtered_power_array, filtered_time_array = filter_array(polyfit(get_datetime(df)), get_datetime(df), 100000, 0)
    
    filtered_power_array = change_w_to_kw(filtered_power_array)
    
    max_power = max(filtered_power_array)
    
    # I have no idea what is this for
    MAX_COMPLIANCE = 0
    best_vw_limit = 248
    BEST_TOTAL_POINTS = 1

    # Determine which data points are of interest for compliance by comparing actual output vs polyfit predicted output, and voltage conditions
    # Ie. W-Watt curtailment can only occur when P_modelled > P_max_allowed.
    compliance_count, volt_array_compliance, time_array_compliance, absolute_watt_array_compliance, relative_watt_array_compliance, successfull_relative_watt_array, successful_volt_array = determine_compliance(
        polyfit, df, dc_cap_w, 248)
    max_volt_watt_time_array, max_volt_watt_power_array = get_max_volt_watt_curve(dc_cap_w, df, 249)


    if len(volt_array_compliance) > 0:
        return volt_array_compliance, relative_watt_array_compliance, filtered_time_array, filtered_power_array
        # I have no idea what is the use of 
        # compliance_count, volt_array_compliance, time_array_compliance, absolute_watt_array_compliance, relative_watt_array_compliance, successfull_relative_watt_array, successful_volt_array

    return None, None, None, None


# INTEGRATE POWER OUTPUT DATA OVER EACH DAY FOR COMPARISON WITH CURTAILMENT CALCUALTIONS
def determine_total_energy_yields(month, monthly_data, site_organiser):
    """Calculate the energy yield from the power data. 

    Args:
        month (str) : month
        monthly_data (df) : D-PV time series monthly data
        site_organiser(dict) : dictionary of site

    Returns:
        -
    
    Side effect:
        calculating the energy yield and store the value in the total_energy_yield_dict for each c_id
        in the corresponding month. 
    """
    
    count = 0
    for site in site_organiser.values():
        
        for c in site.c_id_data.values():
            if c.c_id not in total_energy_yield_dict.keys():
                total_energy_yield_dict[c.c_id] = {}
            count += 1
            print("count: " + str(count))
            total_energy_yield_dict[c.c_id][month] = calculate_months_energy_yield(c.c_id, monthly_data)
            
def calculate_months_energy_yield(c_id, monthly_data):
    """Itegrate power output data over each day for comparison with curtailment calcualtions.

    Args:
    c_id (int): c_id
    monthly_data (df) : D-PV Time Series Data of a certain site, must contain power and time in utc with certain format

    Returns:
    measured_energy (float) : Amount of energy generation in that month in kWh
    
    Functions needed:
    - area_under_curve

    May be applicable for daily application if the monthly data is already filtered into only certain date.
    """
    
    c_data = monthly_data.loc[monthly_data['c_id'] == c_id]

    c_data['utc_tstamp'] = c_data.apply(lambda row: remove_tstamp_ms(row['utc_tstamp']), axis=1)
    
    c_data = c_data.sort_values('utc_tstamp', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='first', ignore_index=False, key=None)

    power_data = c_data.power.tolist()
    time_data = c_data.utc_tstamp.tolist()
    measured_energy = area_under_curve(time_data, power_data)/1000
    return measured_energy  

# REMOVING MILISECOND VALUE IN TIMESTAMP STRINGS
def remove_tstamp_ms(tstamp_string):
    """Remove milisecond value in timestamp strings bcs it is not necessary. 

    Args:
    tstamp_string (str) : the tstamp string in the D-PV data. 

    Returns:
    (str) : the tstamp string with the milisecond value removed. 
    """
    
    TIME_FORMAT_1 = "%Y-%m-%d %H:%M:%S.%f"
    TIME_FORMAT_2 = "%Y-%m-%d %H:%M:%S"
    x = datetime.strptime(tstamp_string, TIME_FORMAT_1)
    return x.strftime(TIME_FORMAT_2)

def area_under_curve(time_data, power_data):
    """Integrate the power curve in power-time curve to get the energy value. 

    Args:
    time_data (list): list of time value
    power_data (list): list of power value

    Returns:
    energy (float) : the value of integration result. 
    """
    
    energy = 0
    
    for i in range(0, len(time_data) - 1):
        t2 = change_to_timestamp(time_data[i+1])
        t1 = change_to_timestamp(time_data[i])
        
        dt = t2-t1
        
        trapArea = (dt / 3600) * 0.5 * (power_data[i] + power_data[i+1])
        energy += trapArea
        
    return energy

def change_to_timestamp(timeString):
    """Convert a string to timestamp type.

    Args:
    timeString (str): time in str

    Returns:
    (dt timestamp): time in timestamp type
    """
    
    element = datetime.strptime(timeString,'%Y-%m-%d %H:%M:%S')
    return datetime.timestamp(element)





def filter_array(x_array, y_array, max_val, min_val):
    """FILTER ARRAY TO INCLUDE VALUES WITHIN A CERTAIN RANGE

    Args:
        x_array (list) : list of the x values
        y_array (list) : list of the y values
        max_val (float) : maximum x value for the filter
        min_val (float) : minimum x value for the filter

    Returns:
        (pd series) : list of filtered x values
        (pd series) : list of filtered y values
    """
    
    filter_arr = []
    for val in x_array:
        if val > max_val or val < min_val:
            filter_arr.append(False)
        else:
            filter_arr.append(True)
    # NOTE: conversion between series and lists was for conveniences of used filter operator, but could be adjusted for better time performance
    x_series = pd.Series(x_array)
    y_series = pd.Series(y_array)

    return x_series[filter_arr].tolist(), y_series[filter_arr].tolist()



def get_datetime(df):
    """TRANSFORM A TIMESTAMP STRING INTO A TIMESTAMP INT VALUE (SECONDS SINCE 1970)
    
    Args:
    df (df) : D-PV time series data with ts column

    Returns:
    datenums (ndarray) : List of float unix timestamp
    
    This is used for polyfit preparation.
    """
    
    dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df.ts]
    datenums = md.date2num(dates)
    return datenums

def change_w_to_kw(filtered_power_array):
    """Convert list of power value from watt to kW unit. 
    
    Args:
    filterdPowerArray (list) : list of power value

    Returns:
    l (list) : list of power value with unit of kW
    """
    
    l = []
    for x in filtered_power_array:
        l.append(x/1000)
        
    return l

# INDIVIDUAL DAY/SITE ANALYSIS 
def determine_compliance(polyfit, graph_df, max_power, vwLimit):
    """Count how many datapoints are in the buffer range of the referenced VW curve.

    Args:
        polyfit (Polyfit) : a function to map timeSeries value to power vlaue
        graph_df (df): time series D-PV data
        maxPOwer (float) : maximum power
        vwLimit (float) : a single value of the vwLimit that we want to investigate. Could be 235-255 V. This
                            is the value where the maximum allowed power starts decreasing.

    Returns:
        compliance_count (int) : number of datapoints in the buffer range of the referenced VW curve
        volt_array (list) : list of voltage value which experience VWatt curtailment 
                            (max expected power > max allowed power)
        time_array (list) : list of time value which experience VWatt curtailment
        absolute_watt_array (list) : list of power value in watt which experience VWatt curtailment
        relative_watt_array (list) : list of cf value (power / inverter power capacity), 0-1 
                                    which experience VWatt curtailment
        successfull_relative_watt_array (list) : list of relative_watt_array in the buffer range
        successful_volt_array (list) : list of volt_array in the buffer range
        
    Functions needed:
        - get_single_date_time
        - volt_watt_curve
    """
    

    volt_array = []
    time_array = []
    absolute_watt_array = []
    relative_watt_array = []

    successfull_relative_watt_array = []
    successful_volt_array = []

    complianceArray = [] #not sure what is this for

    # TODO: Changing to list aided with how analysis functions were created, should be kept as pd series and adjust analysis functions for better time performance
    df_power = graph_df.power.tolist()
    df_time = graph_df.ts.tolist()
    df_voltage = graph_df.voltage.tolist()

    for i in range(len(df_power)):

        actual_power = df_power[i]
        voltage = df_voltage[i]
        timestamp = get_single_date_time(df_time[i])

        # Expected power for the time of day
        expected_power = polyfit(timestamp)

        # Expected max power based on volt-watt
        max_vw_power = volt_watt_curve(voltage, vwLimit) * max_power

        # CALCULATING THE AMOUNT OF OBSERVED CURTAILMENT
        if max_vw_power < expected_power:
            volt_array.append(voltage)
            time_array.append(timestamp)

            absolute_watt_array.append(actual_power)
            relative_watt_array.append(actual_power / max_power)

    # Perform compliance count
    compliance_count = 0
    #I am not really sure about this, bcs in Tim's thesis the buffer is simply 0.07 kW for both
    #the upper and the lower buffer
    BUFFER_HIGH_VALS = 0.03 * 1000 
    BUFFER_LOW_VALS = 0.09 * 1000

    # I have no ide why Tim marks below's code as a comment. 
    # for i in range(len(relative_watt_array)):
    #
    #     relative_watt = relative_watt_array[i]
    #     expected_watt = volt_watt_curve(volt_array[i], vwLimit)
    #
    #     if relative_watt > 0.9:
    #         if expected_watt - BUFFER_HIGH_VALS < relative_watt < expected_watt + BUFFER_HIGH_VALS:
    #             compliance_count += 1
    #             successfull_relative_watt_array.append(relative_watt)
    #             successful_volt_array.append(volt_array[i])
    #
    #     else:
    #         if expected_watt - BUFFER_LOW_VALS < relative_watt < expected_watt + BUFFER_LOW_VALS:
    #             compliance_count += 1

    return compliance_count, volt_array, time_array, absolute_watt_array, relative_watt_array, successfull_relative_watt_array, successful_volt_array

def get_max_volt_watt_curve(max_power, graph_df, vwLimit):
    """RETURNS THE MAXIMUM ALLOWED W/VA AND TIME LIST BASED ON AN INVERTER'S VOLTAGE DATA

    Args:
        max_power (float) : maximum power value
        graph_df (df) : D-PV time series data containing voltage, power, and time col
        vwLimit (value) : voltage value when the maximum allowed power starts decreasing

    Returns:
        max_volt_watt_time_array (list) : list of time
        max_volt_watt_power_array (list) : list of maximum allowed power (in kW) for time in max_volt_watt_time_array
        
    Functions needed:
        - get_single_date_time
        - volt_watt_curve
    """
    
    max_volt_watt_time_array = []
    max_volt_watt_power_array = []

    # TODO: SHOULD BE CHANGED TO A COLUMN WISE FUNCTION FOR BETTER TIME PERFORMANCE
    for row in graph_df.iterrows():
        voltage = row[1].voltage

        max_volt_watt_time_array.append(get_single_date_time(row[1].ts)) #convert to datetime object

        max_volt_watt_power_array.append(volt_watt_curve(voltage, vwLimit) * max_power / 1000) #obtain the max allowed voltage value

    return max_volt_watt_time_array, max_volt_watt_power_array

# GO THROUGH THE COMBINED VW BEHAVIOUR DATA FOR ALL SITES 
def overall_volt_watt_assessment(overall_volt_watt_dict, complaincePercentageLimit, BUFFER_HIGH_VALS, BUFFER_LOW_VALS): #buf 
    """Assess the whole site for VWatt response and count VWatt, Non VWatt, and inconclusive.

    Args:
        overall_volt_watt_dict (dict) : a dict containing all sites in a clear sky day
        complaincePercentageLimit (float) : threshold limit for VWatt response determination
        BUFFER_HIGH_VALS (float) : the amount of upper & lower buffer for the VW curve, used in low W/VA
        BUFFER_LOW_VALS (float) : the amount of upper & lower buffer for the VW curve, used in high W/VA
        
    Returns:
        None, but summarize the VWatt sites, Non VWatt sites, and inconclusive sites count.
        
    Functions needed:
        - site_volt_watt_assessment
    """
    
    best_vw_limit = 248
    
    countVW = 0
    countNVW = 0
    countNA = 0
    
    global site_id_dict
    
    # AGGREGATE RESULTS FOR STATISTICAL ANALYSIS
    for c_id in overall_volt_watt_dict.keys():
        if c_id not in overall_volt_watt_dict.keys():
            continue
        res = site_volt_watt_assessment(c_id, overall_volt_watt_dict[c_id], complaincePercentageLimit, BUFFER_HIGH_VALS, BUFFER_LOW_VALS)
        '''
        if res is None:
            countNA += 1
            buffers_site_id_dict[BUFFER_LOW_VALS]["NA"].append(c_id)
            print("\n!!! NOT ENOUGH POINTS !!!\n")
            
        elif res == True:
            countVW += 1
            buffers_site_id_dict[BUFFER_LOW_VALS]["VW"].append(c_id)
            print("\n!!! VOLT-WATT !!!\n")
            
        elif res == False:
            countNVW += 1
            buffers_site_id_dict[BUFFER_LOW_VALS]["NVW"].append(c_id)
            print("\n!!! NON-VOLT-WATT !!!\n")'''

        if res is None:
            countNA += 1
            site_id_dict["NA"].append(c_id)
            print("\n!!! NOT ENOUGH POINTS !!!\n")

        elif res == True:
            countVW += 1
            site_id_dict["VW"].append(c_id)
            print("\n!!! VOLT-WATT !!!\n")

        elif res == False:
            countNVW += 1
            site_id_dict["NVW"].append(c_id)
            print("\n!!! NON-VOLT-WATT !!!\n")
    
    totalSites = countVW + countNVW 
    
    
    if totalSites == 0: totalSites = 1
    print("FOR4 buffer: " + str(BUFFER_LOW_VALS))
    print("\n\nVolt-Watt sites: " + str(countVW) + " = " + str2(countVW/totalSites*100) + "%")
    print("NON Volt-Watt sites: " + str(countNVW) + " = " + str2(countNVW/totalSites*100) + "%")
    print("Not enough points to assess: " + str(countNA))
    print("Total sites: " + str(countNA + totalSites))
    
def str2(num):
    """Round float to 2 decimal points and then convert it into string type.

    Args:
        num (float): a number
        
    Returns:
        (str) : a number with 2 decimal points
    """
    
    return str(round(num, 2))


def display_day(c_id, date, df, dc_cap_w, volt_array, relative_watt_array, filtered_time_array, filtered_power_array):
    """Display both power/voltage vs time plot, and W/VA vs voltage

    Args:
        c_id (int) : circuit id
        date (str) : date
        df (df) : D-PV time series data
        dc_cap_w (int) : AC Capacity of the inverter
        volt_array (list) : list of voltage values
        relative_watt_array (list): list of actual power corresponding to time list
        filtered_time_array (list): list of time stamp 
        filtered_power_array (list) : list of actual power filtered to include only when expected power is higher than
                                    power limit due to VWatt curve.
        
    Returns:
        None
    
    Functions needed:
        - get_max_volt_watt_curve
        - get_sample_voltages
        - get_watts_curve
        - get_watts_curve_buffer
    
    Side effects:
        Show 2 plots. First plot is time series plot, second plot is power vs voltage plot. 
    """
    
    # Returns the maxmimum permitted real power output based on the inverter's voltage conditions
    max_volt_watt_time_array, max_volt_watt_power_array = get_max_volt_watt_curve(dc_cap_w, df, 250)
    
    plt.style.use('seaborn-whitegrid')
    plt.subplots_adjust(bottom=0.1)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.grid(False)

    ax.tick_params(axis='y', labelcolor='red')
    lns1 = ax.plot(get_datetime(df), df.voltage, 'tomato', label='Local voltage')
    plt.ylabel("Voltage (V)")

    ax2 = ax.twinx()
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.plot(max_volt_watt_time_array,max_volt_watt_power_array, 'limegreen')
    plt.plot(filtered_time_array, filtered_power_array, 'blue')
    
    lns4 = ax2.plot(get_datetime(df), df.power/1000, 'skyblue', label='Real power')
    plt.ylabel("Power (kW)")
    plt.title("c_id: " + str(c_id) + "   Date: " + date + "   DC cap: " + str(dc_cap_w))
    plt.show()
    
    
    plt.title("c_id: " + str(c_id) + "   Date: " + date + "   DC cap: " + str(dc_cap_w))

    z = np.polyfit(volt_array, relative_watt_array, 1)

    slope, intercept = np.polyfit(volt_array, relative_watt_array, 1)
    
    p = np.poly1d(z)
    xRange = list(range(248,260))
    
    plt.plot(xRange,p(xRange),"r--")
    
    plt.plot(get_sample_voltages(230, 266), get_watts_curve(250), label='Best VW limit fit')
    plt.plot(get_sample_voltages(250, 266), get_watts_curve_buffer(250, 0.05), label='Upper buffer')
    plt.plot(get_sample_voltages(250, 266), get_watts_curve_buffer(250, -0.05), label='Lower buffer')

    plt.scatter(volt_array, relative_watt_array, c="purple", marker='.', label='Inverter data')
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (p.u.)")
    plt.show()
    plt.close()
    
def get_sample_voltages(a, b):
    """Return a list of numbers within specified range.

    Args:
        a (int): initial value
        b (int): ending value
        
    Returns:
        (list): a, a+1, a+2, ..., b
    """
        
    return list(range(a, b))

def get_watts_curve(vwLimit):
    """Get an array of voltage and its corresponding vwlimit power. 

    Args:
        vwLimit (int): the voltage value where the actual maximum power starts to decrease.
        
    Returns:
        curve (array): voltage and its corresponding maximum power limit.
        
    Functions needed:
        - get_sample_voltages
        - volt_watt_curve
    """
    
    curve = []
    vs = get_sample_voltages(230, 266) #this seems like useless, only for testing purpose earlier. 
    for v in vs:
        curve.append(volt_watt_curve(v, vwLimit))
    return curve

# PRODUCES V-WATT REDUCTION CURVE FOR A SPECIFIC V-WATT LIMIT WITH A SPECIFIED BUFFER
def get_watts_curve_buffer(vwLimit, buffer):
    """Produces V-Watt reduction curve for a specific V-Watt limit iwth a specified buffer. 

    Args:
        vwLimit (int): the voltage value where the actual maximum power starts to decrease.
        buffer (int): power buffer.
        
    Returns:
        curve (array): voltage and its corresponding maximum power limit.
    """
    
    curve = []
    vs = list(range(vwLimit, 266))
    for v in vs:
        curve.append(min([volt_watt_curve(v, vwLimit) + buffer, 1]))
    return curve

def site_volt_watt_assessment(c_id, site_volt_watt_dict, complaincePercentageLimit, BUFFER_HIGH_VALS, BUFFER_LOW_VALS): #buf
    """Check VWatt behaviour of a certain site from with a certain threshold of number of points in the buffer range.

    Args:
        c_id (int) : c_id
        site_volt_watt_dict (dict) : the key is site_id, the value is a dict with v, p, and d.
        complaincePercentageLimit (float) : threshold limit for VWatt response determination
        BUFFER_HIGH_VALS (float) : the amount of upper & lower buffer for the VW curve, used in low W/VA
        BUFFER_LOW_VALS (float) : the amount of upper & lower buffer for the VW curve, used in high W/VA
  
    Returns:
        (bool) : True if VWatt, False if not VWatt, None if inconclusive due to 
                either not enough point or not enough overvoltage data.
                
    Functions needed:
        determine_volt_watt_scatter_compliance
    """
    
    best_compliance_count = 0
    best_compliance_percentage = 0
    best_vw_limit = None
    best_volt_array = None
    best_relative_watt_array = None
    best_successful_relative_watt_array = None
    best_successful_volt_array = None
        
    # BUFFER AND ANGLE SETTINGS FOR THE ANALYSIS
    COMPLIANCE_COUNT_LIMIT = 150
    TOTAL_POINTS_COUNT_LIMIT = 150
    UPPER_ANGLE_LIMIT = -0.03 #I have no idea what is this for
    LOWER_ANGLE_LIMIT = -0.06 #I have no idea what is this for
    
    # VARIABLE TO CHECK IF THE ANALYSIS RAN OUT OF POINTS AT 256V OR BEFORE. 
    # IF AT 256V AND NO VW BEHAVIOUR IDENTIFIED THEN INCONCLUSIVE RESULT
    NOT_ENOUGH_POINTS_V = 256
    
    print("\n\nc_id: " + str(c_id))
    for vwLimit in list(range(246,258)): #I am not sure why this is 246 until 257. V3 should be 235 to 255.
        compliance_count, compliance_percentage, volt_array, relative_watt_array, successfull_relative_watt_array, successful_volt_array = determine_volt_watt_scatter_compliance(vwLimit, site_volt_watt_dict['v'], site_volt_watt_dict['p'], BUFFER_HIGH_VALS, BUFFER_LOW_VALS)
        if len(volt_array) == 0:
            print("Ran out of points at VWLimit " + str(vwLimit))
            NOT_ENOUGH_POINTS_V = vwLimit
            break
        
        # IF THE RESULT HAS HIGHER COMPLIANCE THAN PREVIOUS V THRESHOLD MEASURE, USE IT INSTEAD
        if best_compliance_count < compliance_count:
            best_compliance_count = compliance_count
            best_vw_limit = vwLimit
            BEST_TOTAL_POINTS = len(volt_array)
            best_volt_array = volt_array
            best_relative_watt_array = relative_watt_array
            best_successful_relative_watt_array = successfull_relative_watt_array
            best_successful_volt_array = successful_volt_array
            best_compliance_percentage = compliance_percentage
           
    
    if best_compliance_count > 0:        
        print("Best VWLimit: " + str(best_vw_limit)) 
        
    else:
        print("No VWLimit results in any compliance")
    
    if best_compliance_count > 0 and BEST_TOTAL_POINTS > TOTAL_POINTS_COUNT_LIMIT:
        slope, intercept = np.polyfit(best_volt_array, best_relative_watt_array, 1)
        print("Slope: " + str(slope))
        
        
        if best_compliance_count > COMPLIANCE_COUNT_LIMIT and best_compliance_percentage > complaincePercentageLimit and LOWER_ANGLE_LIMIT < slope and slope < UPPER_ANGLE_LIMIT:
            return True
        
        else:
            if NOT_ENOUGH_POINTS_V < 256:
                return None
            else:
                return False
    
    else:
        return None

def determine_volt_watt_scatter_compliance(vwLimit, originalVoltArray, originalRelativeWattArray, BUFFER_HIGH_VALS, BUFFER_LOW_VALS):
    """CHECKS EACH DATA POINT TO SEE IF IT FITS WITHIN THE NECESSARY BUFFER TO BE ADDED TO THE SUCCESSFUL DATAPOINT LIST.

    Args:
        vwLimit (float) : Voltage value where the maximum allowed power starts decreasing 
        originalVoltArray (list) : List of all voltage which is curtailed, ie expected power is higher than maximum allowed power
        originalRelativeWattArray (list) : List of all relative power which is curtailed
        BUFFER_HIGH_VALS (float) : the amount of upper & lower buffer for the VW curve, used in low W/VA
        BUFFER_LOW_VALS (float) : the amount of upper & lower buffer for the VW curve, used in high W/VA

    Returns:
        compliance_count (int) : number of points falling in the buffer range
        compliance_percentage (int) : percentage of points falling in the buffer range
        volt_array (list) : filtered voltage value throwing away outlier
        relative_watt_array (list) : filtered relative power
        successfull_relative_watt_array (list) : relative power in the VW curve buffer range
        successful_volt_array (list) : voltage in the VW curve buffer range
        
    Functions needed:
        - filter_array
        - volt_watt_curve
        
    """
    
    compliance_count = 0
    successfull_relative_watt_array = []
    successful_volt_array = []

    # FILTER DATA TO ONLY EXAMINE VALUES HIGHER THAN THE VW LIMIT (AND LOWER THAN 1000, USED AS filter_array FUNCTION IS USED ELSEWHERE)
    volt_array, relative_watt_array = filter_array(originalVoltArray, originalRelativeWattArray, 1000, vwLimit)

    for i in range(len(relative_watt_array)):

        relative_watt = relative_watt_array[i]
        expected_watt = volt_watt_curve(volt_array[i], vwLimit)

        # FOR HIGHER W/VA USE A SMALLER BUFFER, AS THESE VALUES ARE MORE LIKELY TO SUFFER RANDOM VARIATIONS
        if relative_watt > 0.9:
            if expected_watt - BUFFER_HIGH_VALS < relative_watt < expected_watt + BUFFER_HIGH_VALS:
                compliance_count += 1
                successfull_relative_watt_array.append(relative_watt)
                successful_volt_array.append(volt_array[i])

        # FOR LOWER W/VA USE A LARGER BUFFER, AS THESE VALUES ARE LESS LIKELY TO SUFFER RANDOM VARIATIONS
        else:
            if expected_watt - BUFFER_LOW_VALS < relative_watt < expected_watt + BUFFER_LOW_VALS:
                compliance_count += 1
                successfull_relative_watt_array.append(relative_watt)
                successful_volt_array.append(volt_array[i])

    compliance_percentage = 0
    if len(volt_array) > 0:
        compliance_percentage = compliance_count/len(volt_array)
    return compliance_count, compliance_percentage, volt_array, relative_watt_array, successfull_relative_watt_array, successful_volt_array

def assess_curtailment_day(df, c_id, date, dc_cap_w):
    """Quantify curtailment due to V-Watt for sites that have already been identified as having V-Watt enabled.

    Args:
        df (df): D-PV time series data
        c_id (int): circuit id
        date (str): date
        dc_cap_w (int): ac capacity of the inverter

    Returns:
        volt_array_compliance (list): list of voltage values
        relative_watt_array_compliance (list): list of power values in the buffer range of VWatt curve
        filtered_time_array (list): list of filtered time ready for polyfit
        filtered_power_array (list): list of filtered power ready por polyfit
        curtailment/1000 (float): the amount of curtailed energy in kWh
        expected_energy/1000 (float): the amount of energy generated expected in kWh
    
    Funcitons Required:
        filter_power_data
        filter_data_limited_gradients
        get_polyfit
        filter_array
        change_w_to_kw
        get_expected_power
        area_under_curve
        determine_compliance
        get_max_volt_watt_curve
    """

    if df is None:
        return None, None, None, None, None, None

    if len(df.index) == 0:
        return None, None, None, None, None, None
    
    if max(df.power) < 0.3:
        return None, None, None, None, None, None

    df = slice_end_off_df(df)

    df = df.loc[df['power'] > 300]

    if len(df.index) < 20:
        return None, None, None, None, None, None


    power_array, time_array = filter_power_data(df)

    power_array, time_array = filter_data_limited_gradients(power_array, time_array)

    if power_array is None or len(power_array) < 20:
        return None, None, None, None, None, None

    polyfit = get_polyfit(get_datetime_list(time_array), power_array, 2)

    filtered_power_array, filtered_time_array = filter_array(polyfit(get_datetime(df)), get_datetime(df), 100000, 0)

    filtered_power_array = change_w_to_kw(filtered_power_array)
    
    max_power = max(filtered_power_array)

    graph_df = df.loc[df['power'] > 0.1 * max_power]
    power_data = graph_df.power.tolist()
    time_data = graph_df.ts.tolist()
    
    power_expected = get_expected_power(time_data, polyfit)
    
    measured_energy = area_under_curve(time_data, power_data)
    expected_energy = area_under_curve(time_data, power_expected)
    

    curtailment = expected_energy - measured_energy
    if curtailment < 0.01:
        curtailment = 0
    
    MAX_COMPLIANCE = 0
    best_vw_limit = 248
    BEST_TOTAL_POINTS = 1
    compliance_count, volt_array_compliance, time_array_compliance, absolute_watt_array_compliance, relative_watt_array_compliance, successfull_relative_watt_array, successful_volt_array = determine_compliance(polyfit, df, dc_cap_w, 249)
    max_volt_watt_time_array, max_volt_watt_power_array = get_max_volt_watt_curve(dc_cap_w, df, 249)

    
    if len(volt_array_compliance) > 0:
        return volt_array_compliance, relative_watt_array_compliance, filtered_time_array, filtered_power_array, curtailment/1000, expected_energy/1000

    return None, None, None, None, None, None

def get_expected_power(time_data, polyfit):
    """Return the expected power data a specific timestamp according to a given polyfit.
    
    Args:
        time_data (list): list of time data
        polyfit (polyfit): a polyfit function, that map time into expected power.

    Returns:
        expected_power (list): list of expected power using the polyfit.
    """
    
    expected_power = []
    
    for t in time_data:
        expected_power.append(polyfit(get_single_date_time(t)))
        
    return expected_power

def filter_curtailment(df):
    """Take the power data and row number from D-PV time-series data & filter out curtailment. Will be used for polyfit regression.

    Args:
    df (df): Time-series D-PV data with power column and timestamp as an index

    Returns:
    power_array (pd series): filtered power data
    time_array (pd datetime): filtered row number data
    """
    
    max_daily_power = max(df.power)
    if len(df.loc[df['power'] == max_daily_power].index) > 1:
        return None, None
    
    filter_first_half = []
    filter_second_half = []
    power_array = df.power
    x_array = df['x_fit']
    
    halfFlag = True  # True is first half, False is second half
    last_highest_power = 0
    
    for power in power_array:

        # IF power IS GREATER THAN last_highest_power THEN INCLUDE power AND INCREASE last_highest_power
        if power > last_highest_power:
            last_highest_power = power
            filter_first_half.append(True)
        else:
            filter_first_half.append(False)

        if power == max_daily_power:
            break
            
    last_highest_power = 0
    
    # PERFORM SAME FILTER ON SECOND SIDE OF POWER ARRAY
    for power in power_array.iloc[::-1]:

        if power == max_daily_power:
            break

        if power > last_highest_power:
            last_highest_power = power
            filter_second_half.append(True)
        else:
            filter_second_half.append(False)
            
    # COMBINE TO FILTERED SIDES
    filter_second_half.reverse()
    filter_array = filter_first_half + filter_second_half
    return x_array[filter_array], power_array[filter_array]