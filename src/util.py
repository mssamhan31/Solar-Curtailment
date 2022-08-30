# Util module
# List of useful functions


# Import required things
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
import calendar
import seaborn as sns; sns.set()
import itertools
import datetime
from time import gmtime, strftime
from matplotlib import cm


def calculate_first_derivative_of_variable(input_df, col_name_string) :
    """Pass a df FOR A SINGLE C_ID (with 'power_kW' col!!!) Returns the same df with one new cols: power_kW_first_deriv."""
    # NOTE - blanks are just non existent in the df, so it effectively skips them (i.e. compared the value before and after the blanks, which should be okay generally... may be some problem cases.)
    new_col_name = col_name_string + '_first_deriv'

    input_df['temp'] = input_df[col_name_string]

    # Get power(t+1) - power(t) note that an increase is positive and a decrease is negative.
    power_kW_first_deriv = pd.DataFrame(input_df['temp'].shift(-1) - input_df['temp'], index = input_df.index)
    power_kW_first_deriv = power_kW_first_deriv.rename(columns = {'temp' : new_col_name})
    input_df = pd.concat([input_df, power_kW_first_deriv], axis = 1)

    # input_df['power_kW_processed'] = input_df['power_kW']

    # input_df.loc[input_df['power_kW_processed'] <= power_lower_lim, 'power_kW_processed'] = 0.0

    # # Get power(t+1) - power(t) note that an increase is positive and a decrease is negative.
    # power_kW_first_deriv = pd.DataFrame(input_df['power_kW_processed'].shift(-1) - input_df['power_kW_processed'], index = input_df.index)
    # power_kW_first_deriv = power_kW_first_deriv.rename(columns = {'power_kW_processed' : 'power_kW_first_deriv'})
    # input_df = pd.concat([input_df, power_kW_first_deriv], axis = 1)
    return input_df


def get_penetration_by_postcode(PC_INSTALLS_DATA_FILE_PATH, DWELLINGS_DATA_FILE_PATH, sum_stats_df, output_df):
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
