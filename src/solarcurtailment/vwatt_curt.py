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
# REMOVE SPACES AND CHECK IF VALUE NULL
def string_to_float(string):
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

def days_in_month(month):
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
    

    

def filter_power_data_index(df):
    """Take the time and power data from D-PV time-series data & filter out curtailment. Will be used for polyfit regression.

    Args:
    df (df): Time-series D-PV data with power column and timestamp as an index

    Returns:
    power_array (pd series): filtered power data
    time_array (pd datetime): filtered timestamp data
    
    This function filter outs data point that is decreasing in the first half, and filters out data point that
    is incerasing in the second half. That happens only if there is curtailment. 
    """
    
    max_daily_power = max(df.power)
    if len(df.loc[df['power'] == max_daily_power].index) > 1:
        return None, None
    
    filter_first_half = []
    filter_second_half = []
    power_array = df.power
    time_array = df.index
    
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
    return power_array[filter_array], time_array[filter_array]

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
    Overall_site_organiser
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
    """filter D-PV data for only a certain date.

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
    """filter D-PV data for only a certain date.

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

def slice_end_off_df(df):
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
def filter_power_data(graph_df):
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

def filter_data_limited_gradients(power_array, time_array):
    """Filter the power_array data so it includes only decreasing gradient (so the shape is parabolic)

    Args:
    power_array (pd series): non curtailment filtered power data
    time_array (pd datetime): non curtailment filtered timestamp data

    Returns:
    power_array (pd series): gradient filtered power data
    time_array (pd datetime): gradient filtered timestamp data
    """

    if power_array is None:
        return None, None

    # IN GENERAL ANLGE MUST BE BETWEEN THESE VALUES
    ANGLE_LOWER_LIMIT = 80
    ANGLE_UPPER_LIMIT = 90

    # BUT AFTER 'CONTINUANCE_LIMIT' CONTINUOUS VALUES HAVE BEEN ACCEPTED, THE LOWER ANGLE LIMIT IS RELAXED TO THIS VALUE BELOW
    WIDER_ANGLE_LOWER_LIMIT = 70
    CONTINUANCE_LIMIT = 2

    gradients = []
    timeGradients = []
    power_array = power_array.tolist()
    time_array = time_array.tolist()
    filter_array = []

    n = len(power_array)
    gradientsCompliance = [0] * n

    runningCount = 0

    for i in range(1, n):
        g = abs(math.degrees(math.atan((power_array[i] - power_array[i - 1]) / (
                    get_single_date_time(time_array[i]) - get_single_date_time(time_array[i - 1])))))

        addFlag = False

        if g > ANGLE_LOWER_LIMIT and g < ANGLE_UPPER_LIMIT:
            addFlag = True
            runningCount += 1

        elif runningCount > CONTINUANCE_LIMIT and g > WIDER_ANGLE_LOWER_LIMIT:
            addFlag = True

        else:
            runningCount = 0

        if addFlag:
            gradientsCompliance[i - 1] += 1
            gradientsCompliance[i] += 1

        if g > 85:
            gradients.append(g)
            timeGradients.append(time_array[i])

    if gradientsCompliance[0] == 1 and gradientsCompliance[1] == 2:
        filter_array.append(True)
    else:
        filter_array.append(False)

    for i in range(1, n - 1):
        if gradientsCompliance[i] == 2:
            filter_array.append(True)
        elif gradientsCompliance[i] == 1 and (gradientsCompliance[i - 1] == 2 or gradientsCompliance[i + 1] == 2):
            filter_array.append(True)
        else:
            filter_array.append(False)

    if gradientsCompliance[n - 1] == 1 and gradientsCompliance[n - 2] == 2:
        filter_array.append(True)
    else:
        filter_array.append(False)
    

    power_array = pd.Series(power_array)
    time_array = pd.Series(time_array)

    power_array = power_array[filter_array]
    time_array = time_array[filter_array]

    return power_array, time_array

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

def get_single_date_time(d):
    """CONVERT A SINGLE STRING TIMESTAMP TO DATETIME OBJECTS

    Args:
    d (str): string timestamp

    Returns:
    daetimeobject
    """
    return md.date2num(datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))


def get_polyfit(x_array, y_array, functionDegree):
    """GET POLYFIT OF DESIRED DEGREE, NEED x_array as float, not dt object

    Args:
    x_array (ndarray) : List of float unix timestamp
    y_array (pd Series): List of power value corresponding to x_array time
    functionDegree (int): Degree of polynomial. Quadratic functions means functionDegree = 2

    Returns:
    polyfit (np poly1d): polyfit model result, containing list of the coefficients and the constant.
                        The first, second, and third values are coef of x^2, x, and the constant.
    """
     

    timestamps = x_array
    xp = np.linspace(timestamps[0], timestamps[len(timestamps) - 1], 1000) #IDK what is this for. Seems redudant.
    z = np.polyfit(timestamps, y_array, functionDegree)
    polyfit = np.poly1d(z)

    return polyfit

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

def get_datetime_list(list_to_convert):
    """CONVERT A LIST STRING TIMESTAMP TO DATETIME OBJECTS, THEN CONVERT IT TO FLOAT OF UNIX TIMESTAMPS.
    
    Args:
    list_to_convert (pd Series) : List of time in str. Example can be time_array

    Returns:
    datenums (ndarray) : List of float unix timestamp
    
    This is used for polyfit preparation.
    """
    # 
    dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in list_to_convert]
    datenums = md.date2num(dates)
    return datenums

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

def volt_watt_curve(v, limit):
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
    

def get_max_volt_watt_curve(max_power, graph_df, vwLimit):
    """RETURNS THE MAXIMUM ALLOWED W/VA AND TIME LIST BASED ON AN INVERTER'S VOLTAGE DATA

    Args:
        max_power (float) : maximum power value
        graph_df (df) : D-PV time series data containing voltage, power, and time col
        vwLimit (value) : voltage value when the maximum allowed power starts decreasing

    Returns:
        max_volt_watt_time_array (list) : list of time
        max_volt_watt_power_array (list) : list of maximum allowed power (in kW) for time in max_volt_watt_time_array
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

def check_overvoltage_avail(data_site):
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

def check_energy_curtailed(curtailed_data):
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
    
    
def check_vwatt_response(data_site, ac_cap):
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
        
    TODO: 
    1. Reassess whether it is necessary to determine VWatt using count and gradient threshold
    2. Test for non VWatt sample & inconclusive sample
    """
    
    global best_percentage, best_count, best_Vlimit, vwatt_data
    #for Vlimit in list(range (246, 258)): #This is from Tim. Tim's range is different, which IDK why.
    best_percentage = 0 #initiation
    for Vlimit in list(range (235, 256)):
        #step 1. Make a power limit value based on VW curve
        data_site['power_limit_vw'] = data_site['voltage'].apply(volt_watt_curve, limit = Vlimit) * ac_cap

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
            
    data_site['power_limit_vw'] = data_site['voltage'].apply(volt_watt_curve, limit = best_Vlimit) * ac_cap
            
    #step 4. If the percentage from the previous step is higher than certain limit, we say it shows VWatt response.
    PERCENTAGE_THRESHOLD = 84
    COUNT_THRESHOLD = 30 #Tim uses 150 for a month data, where a month usually consist of around 5 clear sky days.
    #print(best_percentage)
    #print (best_Vlimit)
    if (best_percentage > PERCENTAGE_THRESHOLD) & (best_count > COUNT_THRESHOLD): #Tim uses count threshold and gradient threshold. I am not sure whether it is necessary.
        vwatt_response = 'Yes'
        vwatt_curt_energy = check_energy_curtailed(vwatt_data)
    elif suspect_data['voltage'].max() < 255:
        vwatt_response = 'Inconclusive due to insufficient data points'
        vwatt_curt_energy = float('nan')
    else: #no Vlimit results a good fit in all possible Vlimit value
        vwatt_response = 'None'
        vwatt_curt_energy = 0
            
    return vwatt_response, vwatt_curt_energy

def check_vwatt_curtailment(data_site, date, is_good_polyfit_quality, file_path, ac_cap):
    """Check the vwatt response and amount of curtailment due to vwatt response. 

    Args:
        data_site (df) : D-PV time series data
        date (str) : date
        is_good_polyfit_quality (bool) : whether the certain date is a clear sky day or not
        file_path (str): file path where the data is saved
        ac_cap(int): ac capacity of the inverter value

    Returns:
        data_site (df) : D-PV time series data, probably better to be removed before because redundant
        vwatt_response (str) : Yes, None, or Inconclusive due to insufficient overvoltage datapoint.
        vwatt_curt_energy (float) : The amount of energy curtailed due to V-Watt response. 
    """
    
    #check if clear sky day. This contains redundant steps like making ghi dict for all days etc, can still be improved.
    is_clear_sky_day = check_clear_sky_day(date, file_path) 
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
    is_overvoltage_avail = check_overvoltage_avail(data_site)

    if not is_overvoltage_avail:
        vwatt_response = 'Inconclusive due to insufficient overvoltage datapoint.'
        vwatt_curt_energy = float('nan')
        print('No voltage point over 235 V')
        return data_site, vwatt_response, vwatt_curt_energy

    #check vwatt-response here
    vwatt_response, vwatt_curt_energy = check_vwatt_response(data_site, ac_cap)
    
    return data_site, vwatt_response, vwatt_curt_energy