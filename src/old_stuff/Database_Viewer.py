import sys
import time
import datetime
import numpy as np
from scipy.signal import savgol_filter, butter, lfilter
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, \
    QLineEdit, QLabel, QComboBox, QCheckBox
from comparison_plot import ComparisonWindow
from load_data_db_to_dict import get_selected_data
import pyqtgraph as pg
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
from db_to_induction_file import unpack_files
import json
from BH_Curve_Viewer import BHCurveWindow
import itertools
from tqdm import tqdm
import pickle


def select_database(self):
    self.widgets_dict['load_data_button'].setText('Loading...')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #7481ad;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")

    file_name, _ = QFileDialog.getOpenFileName(self, 'Open Database File', '', 'Induction Files (*.db)')
    self.database = file_name
    #
    if file_name:
        self.widgets_dict['load_data_button'].setText('Loading...')
        # update drop down menues
        unpack_files(self.database, "temp_data_dir")
        with open("temp_data_dir/02_measurement_dict.json", 'r') as file:
            data = json.load(file)
        self.measurement_protocol = data

        sample_ids = []
        yoke_materials = []
        paste = []
        amplitude = []
        waveform = []
        frequency = []
        duration = []
        s_rate = []
        for key in self.measurement_protocol.keys():
            self.widgets_dict['MEAS_ID'].addItem(key)
            sample_ids.append(self.measurement_protocol[key]['Sample_ID'])
            yoke_materials.append(self.measurement_protocol[key]['Yoke'])
            paste.append(self.measurement_protocol[key]['Paste'])
            amplitude.append(self.measurement_protocol[key]['Amplitude'])
            waveform.append(self.measurement_protocol[key]['Waveform'])
            frequency.append(self.measurement_protocol[key]['Frequency'])
            duration.append(self.measurement_protocol[key]['Duration'])
            s_rate.append(self.measurement_protocol[key]['Sampling_Rate'])

        # smaple ids
        for item in set(sample_ids):
            self.widgets_dict['Sample_ID'].addItem(item)
        # yoke_materials
        for item in set(yoke_materials):
            self.widgets_dict['Yoke'].addItem(item)
        # paste
        for item in set(paste):
            self.widgets_dict['Paste'].addItem(item)
        # amplitude
        for item in set(amplitude):
            self.widgets_dict['Amplitude'].addItem(str(item))
        # waveform
        for item in set(waveform):
            self.widgets_dict['Waveform'].addItem(item)
        # frequency
        for item in set(frequency):
            self.widgets_dict['Frequency'].addItem(str(item))
        # duration
        for item in set(duration):
            self.widgets_dict['Duration'].addItem(str(item))
        # s_rate
    #
    self.widgets_dict['load_data_button'].setText('Select Database')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")


def set_button_status(self, nam):
    old_status = self.button_status_dict[nam]
    color_dict = get_color_dict()
    if old_status:
        self.button_status_dict[nam] = False
        self.widgets_dict[nam].setStyleSheet("QPushButton { color: black; }")
    else:
        self.button_status_dict[nam] = True
        self.widgets_dict[nam].setStyleSheet(f"QPushButton {{ color: {color_dict[nam]}; }}")


def find_matching_measurement_ids(mp, all_conditions):
    matching_indices = []
    for index, details in mp.items():
        if all(str(details.get(key)) == value or value == '-' for key, value in all_conditions.items()):
            matching_indices.append(index)
    return matching_indices


def select_measurements_ids(self, name, index):
    combo_boxes = ['Sample_ID', 'Waveform', 'Frequency', 'Amplitude', 'Duration', 'Yoke', 'Paste']
    all_conditions = {}
    for box in combo_boxes:
        all_conditions[box] = self.widgets_dict[box].currentText()
    mp = self.measurement_protocol
    valid_mea_ids = find_matching_measurement_ids(mp, all_conditions)
    self.widgets_dict['MEAS_ID'].clear()
    self.widgets_dict['MEAS_ID'].addItems(valid_mea_ids)


def add_data_selection_dropdowns(self, names, background):
    def vertical_layout_with_dropdown(names):
        layout = QVBoxLayout()
        for name in names:
            layout.addWidget(QLabel(name))
            cb = QComboBox()
            cb.setMinimumWidth(90)
            cb.addItem('-')
            if name != 'MEAS_ID':
                cb.currentIndexChanged.connect(partial(select_measurements_ids, self, name))
            self.widgets_dict[name] = cb
            layout.addWidget(self.widgets_dict[name])
        return layout

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel('Select data range'))
    main_layout.addStretch()
    data_layout = QHBoxLayout()
    for names_ in names:
        data_layout.addLayout(vertical_layout_with_dropdown(names_))
    main_layout.addLayout(data_layout)
    main_widget.setLayout(main_layout)
    main_widget.setStyleSheet(f"background-color: {background};"
                              f"max-height: 140px;")
    return main_widget


def add_data_selection_buttons(self, names, background):
    def vertical_layout_with_dropdown(names):
        layout = QVBoxLayout()
        for nam in names:
            self.widgets_dict[nam] = QPushButton(nam)
            self.widgets_dict[nam].is_toggled = False
            self.widgets_dict[nam].clicked.connect(partial(perform_calculations_and_plot, self, nam))
            layout.addWidget(self.widgets_dict[nam])
        return layout

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel('Plot Data (raw)'))
    main_layout.addStretch()
    data_layout = QHBoxLayout()
    for names_ in names:
        data_layout.addLayout(vertical_layout_with_dropdown(names_))
    main_layout.addLayout(data_layout)
    main_widget.setLayout(main_layout)
    main_widget.setStyleSheet(f"background-color: {background};"
                              f"max-height: 140px;")
    return main_widget


def calc_fft(signal, sampling_rate):
    fft_result = np.fft.fft(signal)
    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1 / sampling_rate)
    magnitude = np.abs(fft_result)
    return frequency, magnitude


def plot_FFT(self):
    current_data = self.current_data_dict
    if current_data is None:
        return
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Left, bottom, width, height
    color_dict = get_color_dict()
    keys = ['V_prim [V]']  # , 'I_prim [A]', 'V_sek1 [V]', 'V_sek2 [V]']
    for key in keys:
        signal = np.squeeze(current_data[key].to_numpy())
        sampling_rate = current_data['measurement_protocol']['Sampling_Rate']
        frequency, magnitude = calc_fft(signal, sampling_rate)
        # Plotting
        ax.plot(frequency, magnitude, color=color_dict[key], label=key)
    # Set the title and labels using the appropriate methods
    ax.set_title("Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    # Enable grid lines
    ax.grid(True)
    # Limit x-axis to positive frequencies only
    ax.set_xlim(-10, sampling_rate / 2 + 10)
    # Display the legend
    ax.legend()
    plt.show()


def calc_flux(self, name):
    # get signal
    if name == 'Φ1 [mWb]':
        V_sek = self.current_data_dict['V_sek1 [V]']  # 'V_sek1 [V]'
    elif name == 'Φ2 [mWb]':
        V_sek = self.current_data_dict['V_sek2 [V]']  # 'V_sek2 [V]'

    # calculate offset voltage
    offset = self.current_data_dict['y_offset']['V_sek1 [V]']

    # get time in [s]
    Time_V_sek_s = V_sek.index.to_numpy() / np.timedelta64(1, 's')
    # U_sek in [V]
    V_sek_V = V_sek.to_numpy() - offset
    # calc flux
    flux = -1 / 5000 * integrate.cumulative_trapezoid(y=V_sek_V, x=Time_V_sek_s, initial=0)  # 5000 turns
    flux = flux * 1_000  # convert to mWb for better Visualization
    # center flux curve
    periods = self.current_data_dict['period_data']['periods']
    delta = np.mean(flux[periods[0][0]: periods[-1][1]])
    flux = flux - delta
    # convert back to df
    flux_df = pd.DataFrame({'flux': flux}, index=V_sek.index)
    return flux_df


def plot_vertical_lines(self, index_column):
    up_flanks = self.current_data_dict['period_data']['intersections_up_flank']
    down_flanks = self.current_data_dict['period_data']['intersections_down_flank']

    # Adding vertical lines at up flank indices, only if they are not already added
    for idx in up_flanks:
        line_name = f"up_flank_{idx}"  # Unique name for each line
        if line_name not in self.plot_references:
            line = pg.InfiniteLine(pos=index_column[idx], angle=90, pen=pg.mkPen('#858585', width=0.5))
            self.plot_widget.addItem(line)
            self.plot_references[line_name] = line

    # Adding vertical lines at down flank indices, only if they are not already added
    for idx in down_flanks:
        line_name = f"down_flank_{idx}"  # Unique name for each line
        if line_name not in self.plot_references:
            line = pg.InfiniteLine(pos=index_column[idx], angle=90, pen=pg.mkPen('#383838', width=0.5))
            self.plot_widget.addItem(line)
            self.plot_references[line_name] = line


def plot_dc_offset(self, name, color_dict):
    prop = 'V_sek1 [V]'
    periods = self.current_data_dict['period_data']['periods']
    data = self.current_data_dict[prop].to_numpy()
    for idx, period in enumerate(periods):
        df_new = self.current_data_dict[prop].copy()
        df_new = df_new.iloc[[period[0], period[1]]]
        df_new.loc[:] = np.mean(data[period[0]:period[1]])
        self.plot_references[f'dc_offset_I_{idx}'] = self.plot_widget.plot(df_new.index, df_new.to_numpy().squeeze(),
                                                                           pen=pg.mkPen(color_dict[prop], width=1))


def plot_phi(self, name, color_dict):
    df = calc_flux(self, name)
    self.plot_references[name] = self.plot_widget.plot(df.index, df.to_numpy().squeeze(),
                                                       pen=pg.mkPen(color_dict[name], width=1))


def plot_standard_button(self, name, color_dict):
    # Plot data and store reference
    if name == 'Temp [°C]':
        pen_width = 2
        y_offset = 0
    else:
        pen_width = 0.5
        y_offset = self.current_data_dict['y_offset'][name]
    plot_vertical_lines(self, self.current_data_dict[name].index.to_numpy())
    self.plot_references[name] = self.plot_widget.plot(self.current_data_dict[name].index,
                                                       self.current_data_dict[name].to_numpy().squeeze() - y_offset,
                                                       pen=pg.mkPen(color_dict[name], width=pen_width))


def plot_overlayed_periods(self, color_dict):
    data_names = ['V_sek1 [V]']  # ['I_prim [A]', 'V_prim [V]', 'V_sek1 [V]', 'V_sek2 [V]']
    periods = self.current_data_dict['period_data']['periods']
    cmap = plt.cm.get_cmap('jet')
    np.random.seed(7)
    colors = [cmap(np.random.random()) for _ in range(len(periods))]
    plt.axhline(0, color='black', linewidth=1)
    for i, name in enumerate(data_names):
        for idx, period in enumerate(periods):
            df = self.current_data_dict[name].iloc[period[0]:period[1]]
            # # correct y-offset
            df = df - self.current_data_dict['y_offset'][name]
            # if i == 0:
            #     plt.plot(df.to_numpy().squeeze(), color=colors[idx], label=f'period {idx + 1}', linewidth=0.3)
            # else:
            #     plt.plot(df.to_numpy().squeeze(), color=colors[idx], linewidth=0.3)
            sampling_rate = self.current_data_dict['measurement_protocol']['Sampling_Rate']
            freq, magn = calc_fft(df.to_numpy().squeeze(), sampling_rate)
            plt.plot(freq, magn, label=str(idx))
    plt.legend()
    plt.show()


def plot_hysteresis_curve(self):
    # Create the plot
    fig, ax = plt.subplots()
    #
    flux = calc_flux(self, 'Φ2 [mWb]').to_numpy()
    I_offset = self.current_data_dict['y_offset']['I_prim [A]']
    theta = 1000 * (self.current_data_dict['I_prim [A]'].to_numpy() - I_offset)
    periods = self.current_data_dict['period_data']['periods']
    for i, p in enumerate(periods):
        ax.plot(theta[p[0]: p[1]], flux[p[0]: p[1]], label=f'period {i}')
    # Move left and bottom spines to zero
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Hide the right and top spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Adjust ticks to show negative and positive values
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set label positions at the border
    ax.xaxis.set_label_coords(1.05, -0.02)  # Coordinates are relative to the plot dimensions
    ax.yaxis.set_label_coords(-0.05, 1.05)

    # Adding axis labels
    ax.set_xlabel("Φ [mWb]")
    ax.set_ylabel("Θ [A]")

    # Add a legend to the plot
    ax.legend()
    plt.show()


def toggle_button(self, name):
    color_dict = get_color_dict()
    button = self.widgets_dict[name]
    if not button.is_toggled:
        button.setStyleSheet(f"QPushButton {{ color: {color_dict[name]}; }}")
        button.is_toggled = True
        return True
    else:
        button.setStyleSheet(f"QPushButton {{ color: '#000000'; }}")
        button.is_toggled = False
        return False


def plt_autocorr(self, name):
    color_dict = get_color_dict()
    data = self.current_data_dict[name]
    arr = self.current_data_dict[name].to_numpy()
    autocorrelation = np.correlate(arr, arr, mode='full')
    # plt.plot(autocorrelation)
    plt.axhline(0, color='b')
    plt.plot(np.diff(autocorrelation))
    plt.show()

    # signal = np.squeeze(self.current_data_dict['V_sek1 [V]'].to_numpy())
    # sampling_rate = self.current_data_dict['measurement_protocol']['Sampling_Rate']
    # frequency, magnitude = calc_fft(signal, sampling_rate)


def calc_BHN_spectrum(self):
    #
    Time_V_sek_s = self.current_data_dict['V_prim [V]'].index.to_numpy()
    #
    periods = self.current_data_dict['period_data']['periods']
    diff = self.current_data_dict['V_prim [V]'].to_numpy()
    # # crop to full periods
    # diff = diff[periods[0][0]: periods[-1][1]]
    # Time_V_sek_s = Time_V_sek_s[periods[0][0]: periods[-1][1]]
    diff = diff * diff

    # butterworth filter
    # bb, aa = butter(N=5, Wn=[50, 60000], fs=250000, btype='band')
    # diff = lfilter(bb, aa, diff)

    # calc savgol
    # print('calc savgol')
    # diff = savgol_filter(diff, window_length=10000, polyorder=2)
    # plot

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    diff = moving_average(diff, 50000) * 5000

    color_dict = get_color_dict()
    self.plot_references['V_sek_diff'] = self.plot_widget.plot(Time_V_sek_s, diff.squeeze(),
                                                               pen=pg.mkPen(color_dict['V_prim [V]'], width=1))


def perform_calculations_and_plot(self, name):
    # check if data is available
    if self.current_data_dict is None:
        return
    color_dict = get_color_dict()

    # plot FFT
    if name == 'FFT':
        plot_FFT(self)

    # plot BHN
    if name == 'autocorr_I':
        calc_BHN_spectrum(self)

    # plot FFT
    if name == 'auto-compare':
        self.autoCompareWindow.show()

    # plot Hysteresis Curve
    if name == 'Hysteresis':
        meas_id_combobox = self.widgets_dict['MEAS_ID']
        items = [meas_id_combobox.itemText(i) for i in range(meas_id_combobox.count())]

        # get all specimen ids
        samples_dict = {}
        for item in items:
            if item == '-':
                continue
            if self.measurement_protocol[item]['Sample_ID'][2] == 'N':          # select measurements
                if self.measurement_protocol[item]['Sample_ID'] in samples_dict:
                    samples_dict[self.measurement_protocol[item]['Sample_ID']]['meas_id'].append(item)
                else:
                    samples_dict[self.measurement_protocol[item]['Sample_ID']] = {'meas_id': [], 'freq': [], 'ampl': []}
                    samples_dict[self.measurement_protocol[item]['Sample_ID']]['meas_id'].append(item)

        for key_ in samples_dict.keys():
            samp = samples_dict[key_]
            for m_id in samp['meas_id']:
                samp['ampl'].append(self.measurement_protocol[m_id]['Amplitude'])
                samp['freq'].append(self.measurement_protocol[m_id]['Frequency'])
            samp['ampl'] = np.array(samp['ampl'])
            samp['freq'] = np.array(samp['freq'])

        # get unique freqs and ampls
        first_field = samples_dict[next(iter(samples_dict))]
        unique_ampl = np.unique(first_field['ampl'])
        unique_freq = np.unique(first_field['freq'])

        # loop through parameters and create parameter matrix
        parameter_matrix = np.empty((len(unique_ampl), len(unique_freq), len(samples_dict.keys())), dtype=object)
        for ii, amplitude_ in enumerate(unique_ampl):
            for jj, frequency_ in enumerate(unique_freq):
                for kk, specimen_ in enumerate(samples_dict.keys()):
                    # load new data
                    self.widgets_dict['Frequency'].setCurrentText(str(frequency_))
                    self.widgets_dict['Amplitude'].setCurrentText(str(amplitude_))
                    self.widgets_dict['Sample_ID'].setCurrentText(str(specimen_))
                    load_measured_data(self)
                    # plot_hysteresis_curve(self)
                    data = {}
                    data['Φ1 [mWb]'] = np.squeeze(calc_flux(self, 'Φ1 [mWb]').to_numpy())
                    data['Φ2 [mWb]'] = np.squeeze(calc_flux(self, 'Φ2 [mWb]').to_numpy())
                    data['I_offset'] = self.current_data_dict['y_offset']['I_prim [A]']
                    data['I_prim [A]'] = self.current_data_dict['I_prim [A]'].to_numpy()
                    data['periods'] = self.current_data_dict['period_data']['periods']
                    data['Meas_ID'] = self.current_data_dict['measurement_protocol']['Meas_ID']
                    data['measurement_protocol'] = self.current_data_dict['measurement_protocol']
                    curve_parameters = self.bhCurveWindow.addData(data)
                    parameter_matrix[ii, jj, kk] = curve_parameters
                    # self.bhCurveWindow.show()
        evaluation = {}
        evaluation['amplitudes'] = unique_ampl
        evaluation['frequencys'] = unique_freq
        evaluation['sample_ids'] = list(samples_dict.keys())
        evaluation['data'] = parameter_matrix

        with open('SJB_N_Data.pkl', 'wb') as file:
            pickle.dump(evaluation, file)

    # plot Φ
    if name == 'Φ1 [mWb]' or name == 'Φ2 [mWb]':
        button_state = toggle_button(self, name)
        if button_state:
            plot_phi(self, name, color_dict)
        else:
            self.plot_widget.removeItem(self.plot_references[name])

    # plot DC-Offset_I
    if name == 'DC-Offset_I':
        button_state = toggle_button(self, name)
        if button_state:
            plot_dc_offset(self, name, color_dict)
        else:
            # Iterate over a list of keys to avoid RuntimeError for changing dict size during iteration
            for key in list(self.plot_references.keys()):
                if key.startswith('dc_offset_I'):
                    self.plot_widget.removeItem(self.plot_references[key])

    # plot overlayed_periods
    if name == 'overlay_periods':
        button_state = toggle_button(self, name)
        if button_state:
            plot_overlayed_periods(self, color_dict)

    # standard buttons
    keys = ['V_prim [V]', 'I_prim [A]', 'V_sek1 [V]', 'V_sek2 [V]', 'Temp [°C]']
    if name in keys:
        button_state = toggle_button(self, name)
        if button_state:
            plot_standard_button(self, name, color_dict)
        else:
            self.plot_widget.removeItem(self.plot_references[name])
            del self.plot_references[name]


def add_data_calculation_buttons(self, names, background):
    def vertical_layout_with_dropdown(names):
        layout = QVBoxLayout()
        for nam in names:
            self.widgets_dict[nam] = QPushButton(nam)
            self.widgets_dict[nam].is_toggled = False
            self.widgets_dict[nam].clicked.connect(partial(perform_calculations_and_plot, self, nam))
            layout.addWidget(self.widgets_dict[nam])
        return layout

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel('Plot Data (calculations)'))
    main_layout.addStretch()
    data_layout = QHBoxLayout()
    for names_ in names:
        data_layout.addLayout(vertical_layout_with_dropdown(names_))
    main_layout.addLayout(data_layout)
    main_widget.setLayout(main_layout)
    main_widget.setStyleSheet(f"background-color: {background};"
                              f"max-height: 140px;")
    return main_widget


def select_data_by_predefined_param_widget(self):
    names = [('MEAS_ID', 'Sample_ID'), ('Waveform', 'Frequency'), ('Amplitude', 'Duration'), ('Yoke', 'Paste')]
    widget = add_data_selection_dropdowns(self, names=names, background='#7682a6')
    return widget


def select_raw_measured_data(self):
    names = (('V_prim [V]', 'I_prim [A]'), ('V_sek1 [V]', 'V_sek2 [V]'), ('Temp [°C]',))
    widget = add_data_selection_buttons(self, names=names, background='#6b748f')
    return widget


def select_calculated_data(self):
    names = [('Φ1 [mWb]', 'Φ2 [mWb]'), ('DC-Offset_I', 'overlay_periods'), ('FFT', 'Hysteresis'),
             ('auto-compare', 'autocorr_I')]
    widget = add_data_calculation_buttons(self, names=names, background='#3e4e69')
    return widget


def load_db_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['load_data_button'] = QPushButton('Select Database')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")
    self.widgets_dict['load_data_button'].clicked.connect(lambda: select_database(self))
    main_layout.addWidget(self.widgets_dict['load_data_button'])
    return main_layout


def get_button_background_color(button):
    style = button.styleSheet()
    properties = [prop.strip() for prop in style.split(';') if 'background-color' in prop]
    color = properties[0].split(':')[-1].strip() if properties else 'No color set'
    return color


def show_data_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['show_data_button'] = QPushButton('Load Data')
    self.widgets_dict['show_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['show_data_button'].clicked.connect(lambda: load_measured_data(self))
    #
    main_layout.addWidget(self.widgets_dict['show_data_button'])
    return main_layout


def make_save_data_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['save_data_button'] = QPushButton('Export Data')
    self.widgets_dict['save_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['save_data_button'].clicked.connect(lambda: save_data(self))
    main_layout.addWidget(self.widgets_dict['save_data_button'])
    return main_layout


def make_comparison_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    # button 1
    self.widgets_dict['comparison_button'] = QPushButton('Add to Comparison')
    self.widgets_dict['comparison_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 30px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['comparison_button'].clicked.connect(lambda: add_to_comparison(self))
    # button 2
    self.widgets_dict['comparison_clear_button'] = QPushButton('Clear Comparison')
    self.widgets_dict['comparison_clear_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 30px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['comparison_clear_button'].clicked.connect(lambda: clear_comparison(self))
    # button 3
    self.widgets_dict['comparison_save_button'] = QPushButton('Save Comparison')
    self.widgets_dict['comparison_save_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 30px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['comparison_save_button'].clicked.connect(lambda: save_comparison(self))
    # button 4
    self.widgets_dict['auto_comparison_button'] = QPushButton('Auto Comparison')
    self.widgets_dict['auto_comparison_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 30px;}
                                                    QPushButton:pressed {background-color: #7481ad;}  """)
    self.widgets_dict['auto_comparison_button'].clicked.connect(lambda: auto_comparison(self))
    #
    main_layout.addWidget(self.widgets_dict['comparison_button'])
    main_layout.addWidget(self.widgets_dict['comparison_clear_button'])
    main_layout.addWidget(self.widgets_dict['comparison_save_button'])
    main_layout.addWidget(self.widgets_dict['auto_comparison_button'])
    return main_layout


def get_color_dict():
    dct = {}
    dct['V_prim [V]'] = '#b7b6f0'
    dct['I_prim [A]'] = '#cc9587'
    dct['V_sek1 [V]'] = '#76d649'
    dct['V_sek2 [V]'] = '#d6a249'
    dct['Temp [°C]'] = '#7a120b'
    dct['Φ1 [mWb]'] = '#f5914e'
    dct['Φ2 [mWb]'] = '#f77c2a'
    dct['Hysteresis'] = '#347899'
    dct['DC-Offset_I'] = '#994352'
    dct['overlay_periods'] = '#7bd491'
    dct['mmf'] = '#42f5ef'
    dct['auto-compare'] = '#42f5ef'
    dct['autocorr_I'] = '#42f5ef'
    return dct


def load_measured_data(self):
    self.plot_widget.clear()
    # Add a horizontal line at y=0
    line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('white', width=1))
    self.plot_widget.addItem(line)
    self.current_data_dict = get_selected_data(self)


def save_data(self):
    V_prim = self.current_data_dict['V_prim [V]']
    I_prim = self.current_data_dict['I_prim [A]']
    V_sek1 = self.current_data_dict['V_sek1 [V]']
    V_sek2 = self.current_data_dict['V_sek1 [V]']
    Temp = self.current_data_dict['Temp [°C]']
    Temp.index = Temp.index.round('s')
    Time = pd.Series(V_prim.index.total_seconds(), index=V_prim.index, name='Time[s]')
    combined_df = pd.concat([Time, V_prim, I_prim, V_sek1, V_sek2, Temp], axis=1)
    combined_df.to_csv(f"{self.current_data_dict['measurement_protocol']['Meas_ID']}_data.csv", index=False)


def add_to_comparison(self):
    data = {}
    for ref in self.plot_references.keys():
        if not ('flank' in ref):
            data[ref] = {}
            data[ref]['x_values'] = self.plot_references[ref].xData
            data[ref]['y_values'] = self.plot_references[ref].yData
            data[ref]['linewidth'] = self.plot_references[ref].opts['pen'].width()
            data[ref]['Meas_ID'] = self.current_data_dict['measurement_protocol']['Meas_ID']
    self.comparisonWindow.addData(data)
    self.comparisonWindow.update()
    self.comparisonWindow.show()


def clear_comparison(self):
    self.comparisonWindow.clearData()


def save_comparison(self):
    dtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self.comparisonWindow.saveData(f'C:\\Users\\l.hoefler\\Desktop\\exportet_plots\\compare_{dtime}', 'png')


def get_comparison_dict(filename):
    comp_dict = {}
    in_block = False
    current_block_name = None
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith('#'):
                in_block = True
                current_block_name = line.strip()[2::]
                comp_dict[current_block_name] = {}
            elif not line.strip():
                in_block = False
            elif in_block:
                data = line.strip().split('\t')
                comp_dict[current_block_name]['Waveform'] = data[0]
                comp_dict[current_block_name]['Amplitude'] = data[1]
                comp_dict[current_block_name]['Yoke'] = data[2]
                comp_dict[current_block_name]['SampleID'] = data[3]
                comp_dict[current_block_name]['Frequency'] = data[4]
                comp_dict[current_block_name]['Duration'] = data[5]
                comp_dict[current_block_name]['Paste'] = data[6]
    return comp_dict


def make_comparison(self, block):
    vary_vars = [key for key, value in block.items() if value == '-']
    # get options for all those variables
    vary_vars_dict = {}
    for var in vary_vars:
        vary_vars_dict[var] = list({inner_dict[var] for inner_dict in self.measurement_protocol.values()})
    for combination in itertools.product(*args):
        print
        combination


def auto_comparison(self):
    file_name, _ = QFileDialog.getOpenFileName(self, 'Load auto Comparison File', '', 'auto compare file (*.txt)')
    compare_dict = get_comparison_dict(file_name)
    for key in compare_dict.keys():
        make_comparison(self, compare_dict[key])


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data-Visualizer')

        # Set up the main layout
        main_layout = QVBoxLayout()

        # add the comparison plot window
        self.comparisonWindow = ComparisonWindow()
        self.bhCurveWindow = BHCurveWindow(self.comparisonWindow)

        # define widget dict
        self.widgets_dict = {}
        self.plot_references = {}
        self.database = None

        # Add widgets to the layout
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addLayout(load_db_layout(self))
        top_bar_layout.addWidget(select_data_by_predefined_param_widget(self))
        top_bar_layout.addLayout(show_data_layout(self))
        top_bar_layout.addWidget(select_raw_measured_data(self))
        top_bar_layout.addWidget(select_calculated_data(self))
        top_bar_layout.addLayout(make_comparison_layout(self))
        top_bar_layout.addLayout(make_save_data_layout(self))

        # database load button
        main_layout.addLayout(top_bar_layout)

        # Set up the plot
        self.plot_widget = pg.PlotWidget(self)
        main_layout.addWidget(self.plot_widget)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Maximize the window
        self.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()  # This line has been changed to open in full screen.
    sys.exit(app.exec())
