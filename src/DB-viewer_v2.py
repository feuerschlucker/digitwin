import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, \
    QPushButton, QWidget, QFileDialog, QLabel, QComboBox
from load_db import get_selected_data
import pyqtgraph as pg
import pandas as pd
import sqlite3
from PySide6.QtCore import Qt


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data-Visualizer')
        # Set up the main layout
        main_layout = QVBoxLayout()

        # define widget dict
        self.widgets_dict = {}
        # self.plot_references = {}
        self.database = None

        # Add widgets to the layout
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addLayout(load_db_layout(self))
        top_bar_layout.addWidget(laod_widget_group1(self))
        top_bar_layout.addWidget(laod_widget_group2(self))
        top_bar_layout.addWidget(laod_widget_group3(self))
        top_bar_layout.addLayout(load_data_layout(self))

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

def load_db_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['load_db_button'] = QPushButton('Select Database')
    self.widgets_dict['load_db_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 140px;
                                                                        height: 135px;}""")
    self.widgets_dict['load_db_button'].clicked.connect(
        lambda: select_database(self))
    main_layout.addWidget(self.widgets_dict['load_db_button'])
    return main_layout

def laod_widget_group1(self):
    names = [('S_SERIES', "S_NO"),
             ("TIM",  "OPERATOR"), ("ELAPSE", "F_NO")]
    widget = add_data_selection_dropdowns(
        self, names=names, background='#7682a6', label="Select Criteria")
    return widget

def laod_widget_group2(self):
    names = [('MEAS_ID',)]  # Only include MEAS_ID
    widget = add_data_selection_dropdowns(
        self, names=names, background='#7682a6', label="Select Measurement")

    # Adjust the size of the group
    widget.setFixedHeight(50)  # Set smaller height for a single dropdown
    return widget

def laod_widget_group3(self):
    names = [('RAW', "MR0"), ("STF", "STF_D")]
    widget = add_filetype_button(
        self, names=names, background='#7682a6', label="Select Data to be Displayed")
    return widget

def load_data_layout(self):
    # Database loading UI
    main_layout = QVBoxLayout()
    self.widgets_dict['load_data_button'] = QPushButton(
        'Confirm \n Data Selection \n Execute')
    self.widgets_dict['load_data_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
                                                                        border-radius: 4px;
                                                                        max-width: 150px;
                                                                        height: 135px;}""")
    # self.widgets_dict['load_data_button'].clicked.connect(
    #     lambda: build_query(self))
    main_layout.addWidget(self.widgets_dict['load_data_button'])
    self.widgets_dict['load_data_button'].clicked.connect(
        lambda: plot_raw(self))

    return main_layout

def add_data_selection_dropdowns(self, names, background, label):
    def vertical_layout_with_dropdown(names):
        layout = QVBoxLayout()
        for name in names:

            layout.addWidget(QLabel(name))
            cb = QComboBox()
            cb.setMinimumWidth(10)
            cb.setMaximumWidth(120)
            cb.addItem('-')
            self.widgets_dict[name] = cb
            layout.addWidget(self.widgets_dict[name])
        return layout

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel(label))
    main_layout.addStretch()
    data_layout = QHBoxLayout()
    for names_ in names:
        data_layout.addLayout(vertical_layout_with_dropdown(names_))
    main_layout.addLayout(data_layout)
    main_widget.setLayout(main_layout)
    main_widget.setStyleSheet(f"background-color: {background};"
                              f"max-height: 140px;")
    return main_widget

def add_filetype_button(self, names, background, label):
    def vertical_layout_with_buttons(names):
        layout = QVBoxLayout()
        for name in names:

            layout.addWidget(QLabel(name))
            cb = QPushButton()
            cb.setMinimumWidth(10)
            cb.setMaximumWidth(50)
            self.widgets_dict[name] = cb
            layout.addWidget(self.widgets_dict[name])
        return layout

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel(label))
    main_layout.addStretch()
    data_layout = QHBoxLayout()
    for names_ in names:
        data_layout.addLayout(vertical_layout_with_buttons(names_))
    main_layout.addLayout(data_layout)
    main_widget.setLayout(main_layout)
    main_widget.setStyleSheet(f"background-color: {background};"
                              f"max-height: 140px;")
    self.widgets_dict['RAW'].clicked.connect(
        lambda: plot_raw(self))
    self.widgets_dict['STF'].clicked.connect(
        lambda: plot_stf(self, "stf"))
    self.widgets_dict['STF_D'].clicked.connect(
        lambda: plot_stf(self, "stf_d"))
    self.widgets_dict['MR0'].clicked.connect(
        lambda: plot_mr0(self))

    return main_widget

def get_data_for_meas_id(self, meas_id, type):
    if not self.database:
        print("No database selected!")
        return None

    # Connect to the SQLite database
    conn = sqlite3.connect(self.database)

    # Execute a query to get data for the selected MEAS_ID
    query = f"SELECT * FROM {type} WHERE MEAS_ID = ?"
    # Use params to prevent SQL injection
    df = pd.read_sql_query(query, conn, params=(meas_id,))

    conn.close()
    return df

def plot_raw(self):
    # Get the selected MEAS_ID
    selected_meas_id = self.widgets_dict['MEAS_ID'].currentText()
    if selected_meas_id == '-':  # Handle the default/empty case
        print("No MEAS_ID selected!")
        return

    # Fetch data for the selected MEAS_ID
    data = get_data_for_meas_id(self, selected_meas_id, "raw")
    if data is None or data.empty:
        print("No data found for the selected MEAS_ID!")
        return

    # Clear the existing plot and reinitialize settings
    self.plot_widget.clear()  # Clears the plot
    plot_item = self.plot_widget.getPlotItem()
    plot_item.clear()         # Clears PlotItem state
    # plot_item.setLogMode(x=False, y=False)  # Reset log mode

    x = data['Time'].to_numpy()
    y = data['Bit'].to_numpy()
    valid_indices = x > 0
    x = x[valid_indices]
    y = y[valid_indices]

    # Check if x and y are non-empty after filtering
    if x.size == 0 or y.size == 0:
        print("Filtered data is empty. Cannot plot.")
        return

    # Create a new line plot
    line = pg.PlotDataItem(
        x=x,
        y=y,
        pen=pg.mkPen(color='r', width=2)  # Red line, thickness 2
    )
    self.plot_widget.addItem(line)  # Add the line to the plot widget

    # Set axis ranges
    # Add padding for clarity
    plot_item.setXRange(np.min(x), np.max(x), padding=0)
    plot_item.setYRange(np.min(y), np.max(y), padding=0)
    # Set logarithmic scale for x-axis
    plot_item.setLogMode(x=True, y=False)
    # Disable auto-range to prevent rescaling
    plot_item.enableAutoRange('xy', False)
    plot_item.setAutoVisible(False)
    print(f"Plotted data for MEAS_ID: {selected_meas_id}")

def plot_mr0(self):
    # Get the selected MEAS_ID
    selected_meas_id = self.widgets_dict['MEAS_ID'].currentText()
    if selected_meas_id == '-':  # Handle the default/empty case
        print("No MEAS_ID selected!")
        return

    # Fetch data for the selected MEAS_ID
    data = get_data_for_meas_id(self, selected_meas_id, "mr0")
    if data is None or data.empty:
        print("No data found for the selected MEAS_ID!")
        return

    # Clear the existing plot and reinitialize settings
    self.plot_widget.clear()  # Clears the plot
    plot_item = self.plot_widget.getPlotItem()
    plot_item.clear()         # Clears PlotItem state
    # plot_item.setLogMode(x=False, y=False)  # Reset log mode

    x = data['Int1'].to_numpy()
    y = data['Int3'].to_numpy()
    valid_indices = x > 0
    x = x[valid_indices]
    y = y[valid_indices]

    # Check if x and y are non-empty after filtering
    if x.size == 0 or y.size == 0:
        print("Filtered data is empty. Cannot plot.")
        return

    # Create a new line plot
    line = pg.PlotDataItem(
        x=x,
        y=y,
        pen=pg.mkPen(color='y', width=2)  # Red line, thickness 2
    )
    self.plot_widget.addItem(line)  # Add the line to the plot widget

    # Set axis ranges
    # Add padding for clarity
    plot_item.setXRange(np.min(x), np.max(x), padding=0)
    plot_item.setYRange(np.min(y), np.max(y), padding=0)
    # Set logarithmic scale for x-axis
    plot_item.setLogMode(x=True, y=False)
    # Disable auto-range to prevent rescaling
    plot_item.enableAutoRange('xy', False)
    plot_item.setAutoVisible(False)
    print(f"Plotted data for MEAS_ID: {selected_meas_id}")

def plot_stf(self, cum_or_diff):
    # Get the selected MEAS_ID
    selected_meas_id = self.widgets_dict['MEAS_ID'].currentText()
    if selected_meas_id == '-':  # Handle the default/empty case
        print("No MEAS_ID selected!")
        return

    # Fetch data for the selected MEAS_ID
    data = get_data_for_meas_id(self, selected_meas_id, "stf")
    if data is None or data.empty:
        print("No data found for the selected MEAS_ID!")
        return

    # Clear the existing plot and reinitialize settings
    self.plot_widget.clear()  # Clears the plot
    plot_item = self.plot_widget.getPlotItem()
    plot_item.clear()         # Clears PlotItem state
    # plot_item.setLogMode(x=False, y=False)  # Reset log mode

    x = data['SR_ladder'].to_numpy()
    if cum_or_diff == "stf":
        y = data['Cladder_R_ladder'].to_numpy()
        color = "b"
    else:
        y = data['SC_ladder'].to_numpy()
        color = "g"
    valid_indices = y < 1000000
    x = x[valid_indices]
    y = y[valid_indices]

    # Check if x and y are non-empty after filtering
    if x.size == 0 or y.size == 0:
        print("Filtered data is empty. Cannot plot.")
        return

    # Create a new line plot
    line = pg.PlotDataItem(
        x=x,
        y=y,
        pen=pg.mkPen(color=color, width=2)
    )
    self.plot_widget.addItem(line)  # Add the line to the plot widget

    # Set axis ranges
    plot_item.setXRange(np.min(x), np.max(x), padding=0)
    plot_item.setYRange(np.min(y), np.max(y), padding=0)

    # Set logarithmic scale for x-axis
    plot_item.setLogMode(x=False, y=True)
    # Disable auto-range to prevent rescaling
    plot_item.enableAutoRange('xy', False)
    plot_item.setAutoVisible(False)
    print(f"Plotted data for MEAS_ID: {selected_meas_id}")

def build_query(self):
    # Generate a sine curve
    x = np.linspace(0, 2 * np.pi, 100)  # X values
    y = np.sin(x)  # Y values

    # Clear the existing plot
    self.plot_widget.clear()

    # Plot the sine curve
    self.plot_widget.plot(x, y, pen=pg.mkPen(color='yellow', width=2))

    print("Plotted sine curve.")

def load_master(db_filepath):
    conn = sqlite3.connect(db_filepath)
    df = pd.read_sql_query("SELECT * FROM MASTER", conn)
    return df

def select_database(self):
    file_name, _ = QFileDialog.getOpenFileName(
        self, 'Open Database File', '', 'Sqlite3 Database (*.db)')
    print(f"filename :{file_name}")
    self.database = file_name
    #
    if file_name:
        master = load_master(file_name)
        unique_values_dict = {
            col: master[col].unique().tolist() for col in master.columns}

        criteria = ["MEAS_ID", "S_SERIES", "S_NO",
                    "TIM",  "OPERATOR", "ELAPSE", "F_NO"]

        for item in criteria:
            self.widgets_dict[item].addItems(
                list(set(unique_values_dict[item])))

    # self.widgets_dict['load_db_button'].setText('Select Database')
    # self.widgets_dict['load_db_button'].setStyleSheet("""QPushButton {background-color: #c1dae6;
    #                                                                     border-radius: 4px;
    #                                                                     max-width: 150px;
    #                                                                     height: 135px;}""")

def select_measurements_ids(self, name, index):
    combo_boxes = ["MEAS_ID", "S_SERIES", "S_NO",
                   "HEAT_C", "TIM",  "OPERATOR", "ELAPSE", "F_NO"]
    all_conditions = {}
    for box in combo_boxes:
        all_conditions[box] = self.widgets_dict[box].currentText()
    mp = self.measurement_protocol
    valid_mea_ids = find_matching_measurement_ids(mp, all_conditions)
    self.widgets_dict['MEAS_ID'].clear()
    self.widgets_dict['MEAS_ID'].addItems(valid_mea_ids)

def find_matching_measurement_ids(mp, all_conditions):
    matching_indices = []
    for index, details in mp.items():
        if all(str(details.get(key)) == value or value == '-' for key, value in all_conditions.items()):
            matching_indices.append(index)
    return matching_indices

def load_measured_data(self):
    self.plot_widget.clear()
    # Add a horizontal line at y=0
    line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('white', width=1))
    self.plot_widget.addItem(line)
    self.current_data_dict = get_selected_data(self)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.setGeometry(500, 50, 1700, 1200)  # Set x, y, width, height
    main_app.showNormal()
    # main_app.showNormal() # This line has been changed to open in full screen.
    sys.exit(app.exec())
