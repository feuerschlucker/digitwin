import sys
import time
import datetime
import numpy as np
from scipy.signal import savgol_filter, butter, lfilter
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, \
    QLineEdit, QLabel, QComboBox, QCheckBox
from comparison_plot import ComparisonWindow
from load_db import get_selected_data
import pyqtgraph as pg
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from BH_Curve_Viewer import BHCurveWindow
from tqdm import tqdm
import sqlite3
from PySide6.QtCore import Qt


def load_measurement_protocol(db_filepath):
    conn = sqlite3.connect(db_filepath)
    df = pd.read_sql_query("SELECT * FROM MASTER", conn)
    return df


def select_database(self):
    self.widgets_dict['load_data_button'].setText('Loading...')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #7481ad;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")

    file_name, _ = QFileDialog.getOpenFileName(
        self, 'Open Database File', '', 'Sqlite3 Database (*.db)')
    print(f"filename :{file_name}")
    self.database = file_name
    #
    if file_name:
        self.widgets_dict['load_data_button'].setText('Loading...')
        master = load_measurement_protocol(file_name)
        unique_values_dict = {
            col: master[col].unique().tolist() for col in master.columns}

        criteria = ["MEAS_ID", "S_SERIES", "S_NO", "HEAT_C", "TIM",  "OPERATOR", "ELAPSE", "F_NO"]

        for item in criteria:
            print
            self.widgets_dict[item].addItems(
                list(set(unique_values_dict[item])))

    self.widgets_dict['load_data_button'].setText('Select Database')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")


def load_db_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['load_data_button'] = QPushButton('Select Database')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")
    self.widgets_dict['load_data_button'].clicked.connect(
        lambda: select_database(self))
    main_layout.addWidget(self.widgets_dict['load_data_button'])
    return main_layout


def set_button_status(self, nam):
    old_status = self.button_status_dict[nam]
    color_dict = get_color_dict()
    if old_status:
        self.button_status_dict[nam] = False
        self.widgets_dict[nam].setStyleSheet("QPushButton { color: black; }")
    else:
        self.button_status_dict[nam] = True
        self.widgets_dict[nam].setStyleSheet(
            f"QPushButton {{ color: {color_dict[nam]}; }}")


def find_matching_measurement_ids(mp, all_conditions):
    matching_indices = []
    for index, details in mp.items():
        if all(str(details.get(key)) == value or value == '-' for key, value in all_conditions.items()):
            matching_indices.append(index)
    return matching_indices


def select_measurements_ids(self, name, index):
    combo_boxes = ["MEAS_ID", "S_SERIES", "S_NO", "HEAT_C", "TIM",  "OPERATOR", "ELAPSE", "F_NO"]
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
                cb.currentIndexChanged.connect(
                    partial(select_measurements_ids, self, name))
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
            # self.widgets_dict[nam].clicked.connect(partial(perform_calculations_and_plot, self, nam))
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


def plot_vertical_lines(self, index_column):
    up_flanks = self.current_data_dict['period_data']['intersections_up_flank']
    down_flanks = self.current_data_dict['period_data']['intersections_down_flank']

    # Adding vertical lines at up flank indices, only if they are not already added
    for idx in up_flanks:
        line_name = f"up_flank_{idx}"  # Unique name for each line
        if line_name not in self.plot_references:
            line = pg.InfiniteLine(
                pos=index_column[idx], angle=90, pen=pg.mkPen('#858585', width=0.5))
            self.plot_widget.addItem(line)
            self.plot_references[line_name] = line

    # Adding vertical lines at down flank indices, only if they are not already added
    for idx in down_flanks:
        line_name = f"down_flank_{idx}"  # Unique name for each line
        if line_name not in self.plot_references:
            line = pg.InfiniteLine(
                pos=index_column[idx], angle=90, pen=pg.mkPen('#383838', width=0.5))
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
                                                       self.current_data_dict[name].to_numpy(
    ).squeeze() - y_offset,
        pen=pg.mkPen(color_dict[name], width=pen_width))


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


# def add_data_calculation_buttons(self, names, background):
#     def vertical_layout_with_dropdown(names):
#         layout = QVBoxLayout()
#         for nam in names:
#             self.widgets_dict[nam] = QPushButton(nam)
#             self.widgets_dict[nam].is_toggled = False
#             self.widgets_dict[nam].clicked.connect(partial(perform_calculations_and_plot, self, nam))
#             layout.addWidget(self.widgets_dict[nam])
#         return layout

#     main_widget = QWidget()
#     main_layout = QVBoxLayout()
#     main_layout.addWidget(QLabel('Plot Data (calculations)'))
#     main_layout.addStretch()
#     data_layout = QHBoxLayout()
#     for names_ in names:
#         data_layout.addLayout(vertical_layout_with_dropdown(names_))
#     main_layout.addLayout(data_layout)
#     main_widget.setLayout(main_layout)
#     main_widget.setStyleSheet(f"background-color: {background};"
#                               f"max-height: 140px;")
#     return main_widget


def select_data_by_predefined_param_widget(self):
    names = [('MEAS_ID', 'S_SERIES'), ("S_NO", "HEAT_C"),
             ("TIM",  "OPERATOR"), ("ELAPSE", "F_NO")]
    widget = add_data_selection_dropdowns(
        self, names=names, background='#7682a6')
    return widget


def select_raw_measured_data(self):
    names = (('RAW', 'REC'), ('MR0', 'STF'))
    widget = add_data_selection_buttons(
        self, names=names, background='#6b748f')
    return widget


# def select_calculated_data(self):
#     names = [('Φ1 [mWb]', 'Φ2 [mWb]'), ('DC-Offset_I', 'overlay_periods'), ('FFT', 'Hysteresis'),
#              ('auto-compare', 'autocorr_I')]
#     widget = add_data_calculation_buttons(self, names=names, background='#3e4e69')
#     return widget


def get_button_background_color(button):
    style = button.styleSheet()
    properties = [prop.strip() for prop in style.split(';')
                  if 'background-color' in prop]
    color = properties[0].split(
        ':')[-1].strip() if properties else 'No color set'
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
    self.widgets_dict['show_data_button'].clicked.connect(
        lambda: load_measured_data(self))
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
    self.widgets_dict['save_data_button'].clicked.connect(
        lambda: save_data(self))
    main_layout.addWidget(self.widgets_dict['save_data_button'])
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
    Time = pd.Series(V_prim.index.total_seconds(),
                     index=V_prim.index, name='Time[s]')
    combined_df = pd.concat(
        [Time, V_prim, I_prim, V_sek1, V_sek2, Temp], axis=1)
    combined_df.to_csv(
        f"{self.current_data_dict['measurement_protocol']['Meas_ID']}_data.csv", index=False)


def clear_comparison(self):
    self.comparisonWindow.clearData()


def save_comparison(self):
    dtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self.comparisonWindow.saveData(
        f'C:\\Users\\l.hoefler\\Desktop\\exportet_plots\\compare_{dtime}', 'png')


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


# def make_comparison(self, block):
#     vary_vars = [key for key, value in block.items() if value == '-']
#     # get options for all those variables
#     vary_vars_dict = {}
#     for var in vary_vars:
#         vary_vars_dict[var] = list({inner_dict[var] for inner_dict in self.measurement_protocol.values()})
#     for combination in itertools.product(*args):
#         print
#         combination


# def auto_comparison(self):
#     file_name, _ = QFileDialog.getOpenFileName(self, 'Load auto Comparison File', '', 'auto compare file (*.txt)')
#     compare_dict = get_comparison_dict(file_name)
#     for key in compare_dict.keys():
#         make_comparison(self, compare_dict[key])


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
        # top_bar_layout.addWidget(select_calculated_data(self))
        # top_bar_layout.addLayout(make_comparison_layout(self))
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


def keyPressEvent(self, event):
    if event.key() == Qt.Key_Escape:  # Press Escape to exit full screen
        self.showNormal()


def get_close_button(self):
    main_layout = QVBoxLayout()
    self.widgets_dict['close_button'] = QPushButton('Close')
    self.widgets_dict['close_button'].setStyleSheet("""QPushButton {background-color: #e6a9a9;
                                                                    border-radius: 4px;
                                                                    max-width: 150px;
                                                                    height: 60px;}""")
    self.widgets_dict['close_button'].clicked.connect(
        self.close)  # Connect to the close method
    main_layout.addWidget(self.widgets_dict['close_button'])
    return main_layout


def create_task_bar(self):
    layout = QHBoxLayout()
    # layout.addLayout(get_average_button(self))
    # layout.addLayout(get_comparison_button(self))
    layout.addLayout(get_close_button(self))  # Add close button
    return layout


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.setGeometry(100, 100, 800, 600)  # Set x, y, width, height
    main_app.showNormal()
    # main_app.showNormal() # This line has been changed to open in full screen.
    sys.exit(app.exec())
