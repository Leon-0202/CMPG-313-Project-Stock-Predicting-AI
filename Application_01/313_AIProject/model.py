#imports
import os
import pickle
import math
import numpy as numpy
import pandas as pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tkinter.messagebox as messagebox
from datetime import datetime, timedelta

"""Fixed model loading error"""

class Model:
    # variable containing the path to save the model to
    model_name = "model.h5" #changed the file type to h5 to avoid errors
    model_path = os.path.join("Model", model_name)
    
    #variable containing the path to save the scaler to
    scaler_name = "scaler.pkl"
    scaler_path = os.path.join("Model", scaler_name)
    
    timestep_size = 60

    # variable containing the path to save the previous training data to
    # (Ensures that the user will always have a copy of the training data, or analysis won't be able to be done)
    newFileName = "prev_training_data.csv"
    training_data_filepath = os.path.join("Resources", newFileName)
    
    #set plt style
    plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    
    @staticmethod
    def _read_data(filepath):
        #Reads data from a CSV file and returns a pandas DataFrame
        try:
            dataset = pandas.read_csv(filepath)
            return dataset
        except pandas.errors.EmptyDataError:
            messagebox.showerror("Error", "Error: File is empty")
        except FileNotFoundError:
            messagebox.showerror("Error", "Error: File not found")
        except pandas.errors.ParserError:
            messagebox.showerror("Error", "Error: File is not a CSV")
    
    @staticmethod
    def _save_model(model):
        # Saves the provided model to a file named "model.h5" in the project directory
        model.save(Model.model_path)
        
    @staticmethod
    def _save_scaler(scaler):
        #Saves the provided scaler to a file named "scaler.pkl" in the project directory
        with open(Model.scaler_path, "wb") as f:
            pickle.dump(scaler, f)      

    @staticmethod
    def _load_model():
        # Loads a saved model and scaler from the specified pickle file using pickle
        if not os.path.isfile(Model.model_path):
            raise FileNotFoundError("Model file does not exist. Please train the model first.")
        try:
            model = load_model(Model.model_path, compile=False)
            model.compile()
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
    def _load_scaler():
        # Loads a saved scaler from the specified file using pickle
        if not os.path.isfile(Model.scaler_path):
            raise FileNotFoundError("Scaler file does not exist. Please train the model first.")
        try:
            with open(Model.scaler_path, "rb") as f:
                scaler = pickle.load(f)
                return scaler
        except Exception as e:
            raise Exception(f"Error loading scaler: {str(e)}")
        
    @staticmethod
    def _save_training_data(dataset):
        # saves the used training data to a separate file for later use
        try:
            dataset.to_csv(Model.training_data_filepath, index=False)
        except pandas.errors.EmptyDataError:
            messagebox.showerror("Error", "Error: File is empty")
        except FileNotFoundError:
            messagebox.showerror("Error", "Error: File not found")
        except pandas.errors.ParserError:
            messagebox.showerror("Error", "Error: File is not a CSV")
            
    @staticmethod
    def _extract_data(dataset):
        end_date = datetime(2023, 5, 15)  # Desired end date
        years_before = 4 #desired years to go back to
        
        start_date = end_date - timedelta(days=years_before*365)  # Calculate start date by subtracting 4 years
        
        #read the dataset
        dataset['Date'] = pandas.to_datetime(dataset['Date'])  # Convert 'Date' column to datetime type
        mask = (dataset['Date'] >= start_date) & (dataset['Date'] <= end_date)  # Filter dataset between start and end dates
        filtered_dataset = dataset.loc[mask]  # Get the filtered dataset for the last 4 years
        return filtered_dataset
        
    @staticmethod
    def train_model(filepath, depth):
        dataset = Model._read_data(filepath)
        Model._save_training_data(dataset)
        
        dataset = Model._extract_data(dataset) #only uses the past 4 years of data
        
        
        # Train model on Open Stock Price
        data = dataset.filter(['Open'])
        
        #convert dataframe to numpy array
        arrData = data.values
        
        # get number of rows to train the data on
        #want to train on 3 and test on 1 thus we use 0.75
        training_data_len = math.ceil(len(arrData) * 0.75)
        
        # Normalizing the dataset
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled_training_set = scaler.fit_transform(arrData)
        
        #create training dataset
        #create scaled training data set
        train_data = scaled_training_set[0: training_data_len, :]
        
        # Creating X_train and Y_train data structures
        X_train = [] # contains past timestep_size values
        Y_train = [] #contains the timestep+1 size value we want to predict
        
        for i in range(Model.timestep_size, len(train_data)):
            X_train.append(train_data[i-Model.timestep_size:i, 0])
            Y_train.append(train_data[i, 0])
    
        #convert x_train and y_train to numpy arrays for training use for the model
        X_train, Y_train = numpy.array(X_train), numpy.array(Y_train)

        #Reshape the data
        X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build the LSTM Model
        model = Sequential()

        model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam',loss='mean_absolute_error')
        
        # Training the Model
        model.fit(X_train, Y_train, epochs=depth, batch_size=32)
        
        #save the trained model
        Model._save_model(model)
        Model._save_scaler(scaler)
    
    @staticmethod   
    def predict_stock_prices():
        try:           
            #get model
            model = Model._load_model()
            scaler = Model._load_scaler()
            
            #get original dataset
            dataset = Model._read_data(Model.training_data_filepath)
            dataset = Model._extract_data(dataset) #only uses the past 4 years of data
            
            # filter data on Open Stock Price
            data = dataset.filter(['Open'])
        
            #convert dataframe to numpy array
            arrData = data.values
        
            # get number of rows to train the data on
            #want to train on 3 and test on 1 thus we use 0.75
            training_data_len = math.ceil(len(arrData) * 0.75)
            
            # Normalizing the dataset
            scaled_training_set = scaler.fit_transform(arrData)
            
            #create the test dataset
            test_data = scaled_training_set[training_data_len - Model.timestep_size: , :]
            
            X_test = []
            Y_test = arrData[training_data_len:, :]
            
            for i in range (Model.timestep_size, len(test_data)):
                X_test.append(test_data[i-Model.timestep_size:i,0])
    
            #convert data to numpy array
            X_test = numpy.array(X_test)

            #reshape data
            X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            #get model's predicted price values
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            #plot the data
            train = data[:training_data_len]
            valid = data[training_data_len:].copy()
            valid['Predictions'] = predictions
            
            rmse = mean_squared_error(valid['Open'], valid['Predictions'], squared=False)
            mape = mean_absolute_percentage_error(valid['Open'], valid['Predictions'])
            print("RSME: ", rmse)
            print("MAPE: ", mape)
            
            messagebox.showinfo("Success", "Your data has been analyzed successfully!")

                        
            # get all relevant data for outputs and rename
            analysis_results = pandas.concat([dataset['Date'][training_data_len:], valid['Open'], valid['Predictions']], axis=1)
            analysis_results = analysis_results.rename(columns={'Open': 'Actual Price'})
            analysis_results = analysis_results.rename(columns={'Predictions': 'Predicted Price'})
            
            error_data = pandas.DataFrame({'Error Type': ['RMSE:', 'MAPE:'], 'Values': [rmse, mape]})

            return analysis_results, error_data, dataset, train, valid, training_data_len
        
        except pandas.errors.EmptyDataError:
            messagebox.showerror("Error", "Error: File is empty")
        except FileNotFoundError:
            messagebox.showerror("Error", "Error: File not found")
        except pandas.errors.ParserError:
            messagebox.showerror("Error", "Error: File is not a CSV")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            
    @staticmethod
    def showGraph(dataset, train, valid, training_data_len):
        # Convert the "Date" column to datetime type
        dataset['Date'] = pandas.to_datetime(dataset['Date'])

        #visualize the data
        plt.title('Model', fontsize=16)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Open Price USD ($)', fontsize=18)
        plt.plot(dataset['Date'][:training_data_len], train['Open'])
        plt.plot(dataset['Date'][training_data_len:], valid['Open'])
        plt.plot(dataset['Date'][training_data_len:], valid['Predictions'], color='orange')
        plt.legend(['Training data', 'Actual Values', 'Predicted Values'], loc = 'best')
            
        # Rotate and align the x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
            
        # Maximize the plot window
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
