import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import QApplication
from application.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Handwritten Digit Recognition")
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()