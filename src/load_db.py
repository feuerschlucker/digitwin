import sqlite3
import pandas as pd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from time import time


def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        print(f"Execution time: {time() - t1:.6f} seconds")
        return result

    return wrapper


def get_table_data(table_name, conn):
    # Use pandas' read_sql_query function to read the query results into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


def load_measurement_protocol(db_filepath):
    # Connect to your database
    conn = sqlite3.connect(db_filepath)
    measurement_protocol = get_table_data('mp_induction_active', conn)
    measurement_protocol_dict = measurement_protocol.set_index('MEAS_ID').T.to_dict('list')
    conn.close()
    return measurement_protocol_dict


def get_y_offset(array, periods):
    # get data of all available full periods
    array = array[periods[0][0]: periods[-1][1]]
    offset = np.mean(array)
    return offset


@njit(cache=True)
def get_flank_thresh_f(array):
    for i in range(1, len(array)):  # Start from 1, avoiding the check for i == 0
        if (array[i] == 1 or array[i] == 0) and array[i - 1] == 1:
            array[i] = 1
        elif (array[i] == -1 or array[i] == 0) and array[i - 1] == -1:
            array[i] = -1
    return array


def get_intersection_ind(values, indices, slope):
    values_smooth = savgol_filter(values, window_length=int(int(len(values) / 2) / 2) + 1, polyorder=2)
    if slope == 'down':
        return indices[np.argmax(values_smooth <= 0)]
    elif slope == 'up':
        return indices[np.argmax(values_smooth >= 0)]


@decorator_timer
def get_zero_intersections(data):
    # get signal max and min
    max_ = np.percentile(data, 99)  # use percentile to avoid outlier
    min_ = np.percentile(data, 1)  # use percentile to avoid outlier
    # check where signal is greater or lower than 5% of the amplitude
    threshold = np.zeros_like(data)
    threshold[data > max_ * 0.1] = 1
    threshold[data < min_ * 0.1] = -1
    # use this to get the signal flanks
    flank_thres_1 = get_flank_thresh_f(np.copy(threshold))
    flank_thres_2 = get_flank_thresh_f(np.copy(threshold)[::-1])[::-1]
    # data dict
    data_indices = np.arange(data.shape[0])
    up_flanks = np.logical_and(flank_thres_1 == - 1, flank_thres_2 == 1)
    down_flanks = np.logical_and(flank_thres_1 == 1, flank_thres_2 == - 1)
    index_sections_up = np.split(data_indices, np.where(np.diff(up_flanks))[0] + 1)
    index_sections_up = [sect for sect in index_sections_up if up_flanks[sect[0]]]
    index_sections_down = np.split(data_indices, np.where(np.diff(down_flanks))[0] + 1)
    index_sections_down = [sect for sect in index_sections_down if down_flanks[sect[0]]]
    intersections_down = [get_intersection_ind(data[indices], indices, slope='down') for indices in index_sections_down]
    intersections_up = [get_intersection_ind(data[indices], indices, slope='up') for indices in index_sections_up]
    return intersections_up, intersections_down


def get_periods(up_flanks):
    periods = []
    for i in range(len(up_flanks) - 1):
        start = up_flanks[i]
        end = up_flanks[i + 1]
        periods.append((start, end))
    return periods


def get_selected_data(self):
    if self.database is None:
        print('no database selected')
        return

    condition_dict = {}
    if self.widgets_dict['MEAS_ID'].currentText() != '-':
        meas_id = self.widgets_dict['MEAS_ID'].currentText()
    else:
        return

    data_dict = {}
    keys = ['V_prim [V]', 'I_prim [A]', 'V_sek1 [V]', 'V_sek2 [V]']
    df_key = ['ChA', 'ChB', 'ChC', 'ChD']
    data = pd.read_hdf(f"temp_data_dir/{meas_id}_data.h5", key='data')

    # get temp data
    temp_data = pd.read_hdf("temp_data_dir/01_Temp_data.h5", key='data')
    temp_data = temp_data[temp_data['MEAS_ID'] == meas_id]

    # calc relative time
    initial_time = data.index[0]
    data.index = data.index - initial_time
    temp_data.index = temp_data.index - temp_data.index[0]  # Todo time of temp data and measured data not aligned

    # convert data units
    data["ChB"] = data["ChB"] / 1000 * 10  # *1000 mV->V *10 U->I with shunt of 0.1 ohm (primary Current)
    data["ChA"] = data[
                      "ChA"] / 1000 * 10 * -1  # *1000 mV->V *10 from probe setting         (primary Voltage) (-1 because switched connections)
    data["ChC"] = data["ChC"] / 1000 * 10  # *1000 mV->V *10 from probe setting         (secondary Voltage I)
    data["ChD"] = data["ChD"] / 1000 * 10  # *1000 mV->V *10 from probe setting         (secondary Voltage II)

    # add raw measurement data to data dict
    for i, key in enumerate(keys):
        data_dict[key] = data[df_key[i]]
        data_dict[key].name = key
    data_dict['Temp [°C]'] = temp_data['TC10']
    data_dict['Temp [°C]'].name = 'Temp [°C]'

    # get period data (zero intersections of the I_prim signal)
    intersect_up, _ = get_zero_intersections(data["ChB"].to_numpy())

    # get periods starting with up flanks and ending with up flanks
    periods = get_periods(intersect_up)

    # calculate y-offset for every channel
    data_dict['y_offset'] = {}
    for key in keys:
        data_dict['y_offset'][key] = get_y_offset(data_dict[key].to_numpy(), periods)

    # calculate new corrected period data (starting with up flank of offset corrected I_prim)
    intersect_up, intersect_down = get_zero_intersections(data["ChB"].to_numpy() - data_dict['y_offset']['I_prim [A]'])

    # add period data and y-offset data
    data_dict['period_data'] = {'intersections_up_flank': intersect_up,
                                'intersections_down_flank': intersect_down,
                                'periods': get_periods(intersect_up)}

    data_dict['measurement_protocol'] = self.measurement_protocol[meas_id]
    data_dict['measurement_protocol']['Meas_ID'] = meas_id

    return data_dict
