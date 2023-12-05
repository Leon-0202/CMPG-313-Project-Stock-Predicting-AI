#imports
from tkinter import filedialog
import pandas as pandas
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


def _get_dataset():
    dataset = None
    csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    dataset = pandas.read_csv(csv_path)
    return dataset

def _read_data(dataset):
    graphData = None
    existing_run_columns = [col for col in dataset.columns if col.startswith("Predicted Price (run ")]
    
    next_run_index = 1
    
    while f"Predicted Price (run {next_run_index})" in existing_run_columns:
        column = f"Predicted Price (run {next_run_index})"
        graphData = pandas.concat([dataset['Date'], dataset['Actual Price'], dataset[column]], axis=1)
        
        _graph_data(graphData, column, next_run_index)
        
        next_run_index +=1
        
def _graph_data(graphData, column, columnNr):
    
    graphData['Date'] = pandas.to_datetime(graphData['Date'])
    title = f"Model results for run {columnNr}"
    
    # Plotting the Results    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Open Price USD ($)', fontsize=18)
    plt.plot(graphData['Date'], graphData['Actual Price'], color='green')
    plt.plot(graphData['Date'], graphData[column], color='orange')
    plt.legend(['Actual Values', 'Predicted Values'], loc = 'best')
    
    plt.show()
    
    rmse = mean_squared_error(graphData['Actual Price'], graphData[column], squared=False)
    mape = mean_absolute_percentage_error(graphData['Actual Price'], graphData[column])
    print("==========================================================================")
    print(title)
    print("RSME: ", rmse)
    print("MAPE: ", mape)
    print("==========================================================================")
 
dataset = _get_dataset()
_read_data(dataset)
    
    
        

    
    
    
    
    
