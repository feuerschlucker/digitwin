from PySide6.QtWidgets import QMainWindow
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter


def get_default_plot_config():
    dct = {}
    dct['x_limits'] = (-11000, 11000)
    dct['y_limits'] = (-0.9, 0.9)
    dct['x_label'] = 'Θ [A]'
    dct['y_label'] = 'Φ [mWb]'
    dct['x_line'] = True
    dct['y_line'] = True
    dct['grid'] = True
    return dct


class ComparisonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Comparison')
        self.graphWidget = pg.PlotWidget()  # Create a plot widget
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('black')
        self.data = []  # Store plot data
        self.plot_config = get_default_plot_config()
        self.zero_line_h = None
        self.zero_line_v = None

    def update(self):
        if not self.data:
            return
        colors = ['#ff5252', '#e8971c', '#faf861', '#41cc99', '#4ceefc', '#4c78fc', '#3832b3', '#321363', '#f21882']
        color_index = 0
        self.graphWidget.clear()  # Clear existing data and draw everything new
        self.graphWidget.addLegend()

        if self.plot_config['x_line']:  # Todo bugfix everytime a plot is added a new line is drawn?
            self.zero_line_h = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('white', width=1))
            self.graphWidget.addItem(self.zero_line_h)

        if self.plot_config['y_line']:  # Todo bugfix everytime a plot is added a new line is drawn?
            self.zero_line_v = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('white', width=1))
            self.graphWidget.addItem(self.zero_line_v)

        for dat in self.data:
            if dat:
                for key in dat.keys():
                    curve = dat[key]
                    color = colors[color_index % len(colors)]
                    color_index += 1
                    self.graphWidget.plot(curve['x_values'], curve['y_values'], pen=color,
                                          name=f'{curve["Meas_ID"]} ({key})', width=curve['linewidth'])
                    print(key)

        if self.plot_config['x_limits'] is not None:
            x_limits = self.plot_config['x_limits']
            self.graphWidget.plotItem.setXRange(x_limits[0], x_limits[1])

        if self.plot_config['y_limits'] is not None:
            y_limits = self.plot_config['y_limits']
            self.graphWidget.plotItem.setYRange(y_limits[0], y_limits[1])

        if self.plot_config['x_label'] is not None:
            self.graphWidget.setLabel('bottom', self.plot_config['x_label'], color='white', size='12pt')

        if self.plot_config['y_label'] is not None:
            self.graphWidget.setLabel('left', self.plot_config['y_label'], color='white', size='12pt')

        if self.plot_config['grid']:
            self.graphWidget.showGrid(x=True, y=True, alpha=0.3)

    def addData(self, data):
        self.data.append(data)  # Add new data# Re-plot with the new data

    def clearData(self):
        self.data = []
        self.graphWidget.clear()

    def saveData(self, filename, format_):
        if format_ == 'png':
            exporter = ImageExporter(self.graphWidget.plotItem)
            exporter.parameters()['width'] = 1920
            exporter.parameters()['antialias'] = True
            exporter.export(filename + '.png')
