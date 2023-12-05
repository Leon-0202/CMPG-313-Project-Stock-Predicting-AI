"""
Please note the following notation for the sake of consistancy:

#def _function():
    indicates a private function, please do not call these functions outside of the classes they are in.

#def function():
    indicates a public function
    
Please also ensure that you have the following libraries installed:
1. pandas
2. sci-kit-learn
3. tensorflow
4. keras
5. joblib
6. matplotlib
7. yfinance
8. tkcalendar
#Please note that you need to be online to use the program to download data from yfinance
"""

#imports
from gui import GUI

class main:
    # Create an instance of the GUI class and start the program
    gui = GUI()
    gui.run()