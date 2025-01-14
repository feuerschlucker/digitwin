import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QComboBox,
)
from PyQt6.QtCore import Qt, QSize


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Add fields with pull-down menus
        self.add_dropdowns()

        # Replace "Confirm Data Selection Execute" with 2x2 button layout
        self.add_button_grid()

    def add_dropdowns(self):
        # Add pull-down menus (example)
        for i in range(3):
            combo_box = QComboBox()
            combo_box.addItems(["Option 1", "Option 2", "Option 3"])
            self.main_layout.addWidget(combo_box)

    def add_button_grid(self):
        # Create a grid layout for the buttons
        grid_layout = QGridLayout()

        # Add 2x2 quadratic buttons
        for row in range(2):
            for col in range(2):
                button = QPushButton(f"Button {row * 2 + col + 1}")
                #button.setSizePolicy(QPushButton.sizePolicy.expanding, QPushButton.SizePolicy.Expanding)
                button.clicked.connect(self.on_button_click)  # Connect button click
                grid_layout.addWidget(button, row, col)

        # Create a container widget for the grid layout
        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)

        # Add the grid widget to the main layout
        self.main_layout.addWidget(grid_widget)

    def on_button_click(self):
        sender = self.sender()
        print(f"{sender.text()} clicked!")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_button_aspect_ratio()

    def adjust_button_aspect_ratio(self):
        # Ensure buttons are quadratic
        grid_layout = self.central_widget.findChild(QGridLayout)
        if not grid_layout:
            return

        for i in range(grid_layout.count()):
            widget = grid_layout.itemAt(i).widget()
            if widget:
                size = widget.size()
                min_side = min(size.width(), size.height())
                widget.setFixedSize(QSize(min_side, min_side))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(400, 400)  # Set initial window size
    window.show()

    sys.exit(app.exec())
