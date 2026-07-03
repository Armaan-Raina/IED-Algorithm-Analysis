"""Entry point for the IED Detection GUI."""

import sys

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QApplication

from src.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
