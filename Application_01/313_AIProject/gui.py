#imports
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog
from tkcalendar import DateEntry #need to install this
import tkinter.messagebox as messagebox
import os
import threading
import pandas as pandas
import yfinance as yf #need to install this
from datetime import datetime
import matplotlib.pyplot as plt
from model import Model

class GUI:
    
    #global variables
    #path information
    stock_data_path=""
    saved_stock_data_path = os.path.join("Resources", "Downloaded_Stock_Info.csv")
    
    #int
    training_data_len=0
    
    #datasets
    selectable_stock_information = None
    analysis_information = None
    error_data = None
    dataset = None
    train = None
    valid = None
    
    #set graph style
    plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    
    def _create_main_window(self):
        # Create a new window
        window = tk.Tk()

        # Set window attributes
        window.state('zoomed') #makes window fullscreen
        #window.attributes("-topmost", True) #ensures that the window is always on top
        window.resizable(False, False) #ensures that the window is not resizable horizontally/vertically
        window.title("Stocker")
        
        # Create a canvas and add the background image to it
        canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        background_image = self._load_image_icon("background").zoom(2)  # Increase the size of the image by a factor of 2
        canvas.create_image(0, 0, image=background_image, anchor="nw")
        
        # Set window icon
        icon = self._load_image_icon("icon")
        window.iconphoto(True, icon)
        
        #create buttons and other widgets here:
        # Create a content panel
        self._create_content_panel(window)
        
        # Run the main loop
        window.mainloop()
        
    #Use this function to load an image
    def _load_image_icon(self, filename):
        #Load an image icon from a file in the Resources directory.
        path = os.path.join("Resources", filename + ".png")
        return tk.PhotoImage(file=path)
    
    def _create_content_panel(self, window):
        #set the font, style, WrapLength and color for the buttons
        bttnFont = ("Helvetica", 16)
        bttnFg = "#D9D9D9"
        bttnBg = "#353535"
        bttnStyle = "flat"
        wLength = 200
        
        #set panel color
        panelBg="#252729"
        #"#293949"
        
        #set group color and font
        groupFont = ("Helvetica", 12)
        groupBg = "#45494C"
        groupFg = "#FFFFFF"
        
        #set font and color for the labels
        labelFont = ("Helvetica", 14)
        labelFg = "#D9D9D9"
        labelBg = "#45494C"
        
        # Create a panel in the center of the window
        # Create the panel with a dark grey color
        panel = tk.Frame(window, bg=panelBg)
        panel.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.95)
        
        # Create the Download Stock Data group
        download_group = tk.LabelFrame(panel, text="Download Stock Data", font=groupFont, bg=groupBg, fg=groupFg)
        download_group.place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.40)
        
        # Import existing Stock Data group
        import_group = tk.LabelFrame(panel, text="Existing Stock Data", font=groupFont, bg=groupBg, fg=groupFg)
        import_group.place(relx=0.55, rely=0.05, relwidth=0.4, relheight=0.40)
        
        # Download Stock Data comboboxes
        start_date_label = tk.Label(download_group, text="Start Date:", font=labelFont, bg=labelBg, fg=labelFg)
        start_date_label.grid(row=0, column=0, padx=50, pady=10)
        start_date_box = DateEntry(download_group)
        start_date_box.grid(row=0, column=1, padx=10, pady=10, sticky="we")

        end_date_label = tk.Label(download_group, text="End Date:", font=labelFont, bg=labelBg, fg=labelFg)
        end_date_label.grid(row=1, column=0, padx=50, pady=10)
        end_date_box = DateEntry(download_group)
        end_date_box.grid(row=1, column=1, padx=10, pady=10, sticky="we")

        ticker_label = tk.Label(download_group, text="Stock Ticker:", font=labelFont, bg=labelBg, fg=labelFg)
        ticker_label.grid(row=2, column=0, padx=50, pady=10)
        ticker_combobox = ttk.Combobox(download_group)
        ticker_combobox.grid(row=2, column=1, padx=10, pady=10, sticky="we")
        self._fill_combo_box(ticker_combobox)
        
        name_label = tk.Label(download_group, text="Company name:", font=labelFont, bg=labelBg, fg=labelFg)
        name_label.grid(row=3, column=0, padx=50, pady=10)
        namebox = tk.Entry(download_group, state="readonly")
        namebox.grid(row=3, column=1, padx=10, pady=10, sticky="we")
        
        #function to download the stock information
        def on_download_pressed():
            selected_ticker = ticker_combobox.get().upper()
            selected_start_date = start_date_box.get()
            selected_end_date = end_date_box.get()
            comp_name = namebox.get()
            
            if not selected_ticker:
                messagebox.showerror("Error", "Please select a ticker.")
                return

            if not selected_start_date or not selected_end_date:
                messagebox.showerror("Error", "Please select start and end dates.")
                return
            
            start_date = datetime.strptime(selected_start_date, "%m/%d/%y")
            start_date = start_date.strftime("%Y-%m-%d")
            
            end_date = datetime.strptime(selected_end_date, "%m/%d/%y")
            end_date = end_date.strftime("%Y-%m-%d")            
            
            self._download_stock_data(textbox, start_date, end_date, selected_ticker, comp_name)
            
        # Download Stock Data button
        download_button = tk.Button(download_group, text="Download", font=bttnFont, bg=bttnBg, fg=bttnFg, relief=bttnStyle, wraplength=wLength, command=on_download_pressed)
        download_button.grid(row=4, column=0, columnspan=2, padx=50, pady=10)
        
        #create import CSV file button
        import_button = tk.Button(import_group, text="Import Stock Data From CSV", font=bttnFont, bg=bttnBg, fg=bttnFg, relief=bttnStyle, wraplength=wLength, command=lambda: self._import_data_dialog(textbox))
        import_button.grid(row=0, column=0, padx=120, pady=10, sticky="we")
        
        #Create graph button for current data
        visualize_current = tk.Button(import_group, text="Visualize Current Data", font=bttnFont, bg=bttnBg, fg=bttnFg, relief=bttnStyle, wraplength=wLength, command=lambda: self._visualize_current_data(self.stock_data_path))
        visualize_current.grid(row=1, column=0, padx=120, pady=10, sticky="we")
        
        #create export button
        def on_export_pressed():
            if self.analysis_information is None:
                messagebox.showerror("Error", "No analysis data available. Please run an analysis first.")
                return
    
            self._export_options(window)
        
        export_button = tk.Button(import_group, text="Export Prediction To CSV", font=bttnFont, bg=bttnBg, fg=bttnFg, relief=bttnStyle, wraplength=wLength, command=on_export_pressed)
        export_button.grid(row=2, column=0, padx=120, pady=10, sticky="we")
        
        # Textbox to display stock information
        textbox = tk.Text(panel)
        textbox.place(relx=0.05, rely=0.5, relwidth=0.9, relheight=0.35)
        
        # Add a vertical scrollbar to the textbox
        scrollbar = tk.Scrollbar(textbox)
        scrollbar.pack(side="right", fill="y")
        scrollbar.config(command=textbox.yview)
        textbox.config(yscrollcommand=scrollbar.set)
        
        textbox.config(state="disabled") #set the textbox so that it can't be edited

        #error measure text box
        errorbox = tk.Text(panel)
        errorbox.place(relx=0.75, rely=0.875, relwidth=0.2, relheight=0.1)
        errorbox.config(state="disabled") #set the textbox so that it can't be edited
        
        def on_predict_pressed():
            self._train_model(window)
            self._run_analysis(window)
            self._display_data(textbox, self.analysis_information)
            self._display_data(errorbox, self.error_data)
            Model.showGraph(self.dataset, self.train, self.valid, self.training_data_len)
            
        # Predict Stock Prices button
        predict_button = tk.Button(panel, text="Run Data Analysis and Prediction", font=bttnFont, bg=groupBg, fg=bttnFg, relief=bttnStyle, wraplength=wLength, command=on_predict_pressed)
        predict_button.place(relx=0.4, rely=0.875, relwidth=0.2)
        
        # Define the callback function
        def on_ticker_selected(event):
            selected_ticker = ticker_combobox.get().upper()
            self._get_company_name(selected_ticker, namebox)  # Call _get_company_name with the selected ticker

        # Bind the callback function to the combobox's selection event
        ticker_combobox.bind("<<ComboboxSelected>>", on_ticker_selected)
        
    def _import_data_dialog(self, textbox):
        self.stock_data_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        # If a file is selected, display a message box with a success message
        if self.stock_data_path:
            messagebox.showinfo("Success", f"File {self.stock_data_path} has been successfully selected!")
            
            #reset dataset
            self.dataset = None
            
            # Read the data from the CSV file
            try:
                self._read_data(self.stock_data_path, textbox)
                
            except pandas.errors.EmptyDataError:
                messagebox.showerror("Error:", "File is empty")
            except FileNotFoundError:
                messagebox.showerror("Error:", "File not found")
            except pandas.errors.ParserError:
                messagebox.showerror("Error:", "File is not a CSV")
        # If no file is selected, display an error message
        else:
            messagebox.showerror("Error:", "No file selected.")
            
    def _read_data(self, stock_data_path, textbox):
        dataset = pandas.read_csv(self.stock_data_path)
        self._display_data(textbox, dataset)
        
    def _fill_combo_box(self, combobox):
        filename = "tickers"
        filepath = os.path.join("Resources", filename + ".csv")
        # Read the CSV file into a DataFrame
        self.selectable_stock_information = pandas.read_csv(filepath)
    
        # Get the tickers from the DataFrame
        tickers = self.selectable_stock_information['Ticker'].tolist()
    
        # Populate the combobox with tickers
        combobox['values'] = tickers
        
    def _get_company_name(self, ticker, namebox):
        # Get the company name from the DataFrame
        company_name = self.selectable_stock_information.loc[self.selectable_stock_information['Ticker'] == ticker, 'Name'].iloc[0]
        # Update the text box
        namebox.config(state="normal")
        namebox.delete(0, tk.END)
        namebox.insert(0, company_name)
        namebox.config(state="readonly")
            
    def _download_stock_data(self, textbox, startDate, endDate, ticker, comp_name):
        try:
            # Get stock information from API and store it in a dataset
            dataset = yf.download(ticker, start=startDate, end=endDate)

            # Check if dataset is empty
            if dataset.empty:
                messagebox.showerror("Error", "Stock data not available for the selected ticker and date range.")
                return

            # Save the data and set the filepath to this data
            dataset.to_csv(self.saved_stock_data_path, index=True)
            self.stock_data_path = self.saved_stock_data_path

            self._read_data(self.stock_data_path, textbox)

            # Show success message
            messagebox.showinfo("Success", f"Stock data for {comp_name} downloaded successfully.")
            
            #reset dataset
            self.dataset = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download stock data: {str(e)}")
        
    def _display_data(self, textbox, dataset):
        textbox.config(state="normal")
        # Clear the existing content in the textbox
        textbox.delete("1.0", "end")
        # Convert the dataset to a string and insert it into the textbox
        dataset_string = dataset.to_string(index=False)
        # Configure a tag for centered alignment
        textbox.tag_configure("center", justify="center")
        # Insert the centered text into the textbox
        textbox.insert("end", dataset_string, "center")
        textbox.config(state="disabled")
        
    def _visualize_current_data(self, stock_data_path):
        if not stock_data_path:
            messagebox.showerror("Error", "Please import, download or analyze stock data first.")
            return
            
        if not self.dataset is None:
            self._graph_result_data()
        else:
            self._graph_stock_data()

        
    def _train_model(self, window):
        if not self.stock_data_path:
            messagebox.showerror("Error:", "No training file selected. Please first select a file to be used to train the model.")
            return
        
        cancelled = False
        
        # Open dialog window to ask for the depth/nr of epochs
        while True:
            user_input = simpledialog.askstring("Model Training", "Enter the depth/nr of epochs for training:")
            if user_input is None:
                cancelled = True  # User cancelled
            try:
                depth = int(user_input)
                if depth <= 0:
                    raise ValueError("Depth/nr of epochs must be positive")
                break  # Valid input
            except ValueError:
                messagebox.showerror("Error:", "Please enter a valid positive integer value for the depth/nr of epochs.")
                
        if cancelled == True:
            return
        
        # Disable the window and change its title
        try:
            window.title("Please wait... loading...")
            window.config(cursor="wait")
            window.update()
            
            # Calculate the center point of the main window
            center_x = window.winfo_rootx() + window.winfo_width() // 2
            center_y = window.winfo_rooty() + window.winfo_height() // 2

            # Calculate the position of the wait window
            wait_x = center_x - 150  # Adjust the values as needed
            wait_y = center_y - 50
            
            # Create and show the wait window
            wait_window = tk.Toplevel(window)
            wait_window.overrideredirect(True)  # Remove window decorations
            #wait_window.attributes("-topmost", True)  # Set the loading screen window to stay on top of the application window
            
            wait_label = tk.Label(wait_window, text="Please wait while the model is analyzing your data.\nThis will take several minutes...")
            wait_label.pack(padx=10, pady=10)
            wait_window.geometry("+%d+%d" % (wait_x, wait_y))  # Position the window
            wait_window.transient(window)  # Make the loading screen window transient to the application window
            wait_window.lift()
            
            # Add progress bar to wait_window
            progress_bar = ttk.Progressbar(wait_window, mode='indeterminate', length=200)
            progress_bar.pack(padx=5, pady=10)
            progress_bar.start()
            
            # Start a new thread to train the model (Ensures that this runs in the background)
            def train_model_thread():
                Model.train_model(self.stock_data_path, depth)
                wait_window.destroy()
                messagebox.showinfo("Success", "Your model has been trained successfully!\nPlease close this window to run predictions on your data.")

            t = threading.Thread(target=train_model_thread) #start the thread
            t.start()
            
            # Set the original window as the wait_window for the new window (nothing will happen until this window is destroyed)
            window.wait_window(wait_window)
            
        finally:
            # Reset the window title and enable it
            window.config(cursor="")
            window.title("Stocker")
        
    def _run_analysis(self, window):
        if not self.stock_data_path:
            messagebox.showerror("Error:", "No stock file selected. Please first select a file to be analyzed.")
            return
        
        # Run analysis
        # Disable the window and change its title
        try:
            window.title("Please wait... loading...")
            window.config(cursor="wait")
            window.update()
            
            self.analysis_information, self.error_data, self.dataset, self.train, self.valid, self.training_data_len = Model.predict_stock_prices()
            
        finally:
            # Reset the window title and enable it
            window.config(cursor="")
            window.title("Stocker")
            
    def _export_options(self, window):
        export_window = tk.Toplevel(window)
        export_window.title("Export Options")
        export_window.resizable(False, False)
        
        # Calculate the center point of the main window
        center_x = window.winfo_rootx() + window.winfo_width() // 2
        center_y = window.winfo_rooty() + window.winfo_height() // 2

        # Calculate the position of the wait window
        win_x = center_x - 100  # Adjust the values as needed
        win_y = center_y - 50
        
        export_window.geometry("+%d+%d" % (win_x, win_y))  # Position the window
    
        def create_new_csv():
            export_window.destroy()
        
            save_file_dialog = filedialog.asksaveasfile(defaultextension=".csv")
            if save_file_dialog is None:
                return  # User canceled the save dialog
            save_path = save_file_dialog.name
            save_file_dialog.close()
        
            # Prepare data for export
            export_data = self.analysis_information.copy()
            export_data.rename(columns={"Predicted Price": "Predicted Price (run 1)"}, inplace=True)

            # Export to new CSV file
            export_data.to_csv(save_path, index=False)
            messagebox.showinfo("Export Successful", f"Data exported to {save_path} successfully!")

        def merge_with_existing_csv():
            export_window.destroy()

            open_file_dialog = filedialog.askopenfile(filetypes=[("CSV Files", "*.csv")])
            
            if open_file_dialog is None:
                return  # User canceled the open dialog
            open_path = open_file_dialog.name
            open_file_dialog.close()

            # Read existing CSV file
            existing_data = pandas.read_csv(open_path)

            # Check if required columns exist in the existing CSV file
            if "Date" not in existing_data.columns or "Actual Price" not in existing_data.columns or "Predicted Price (run 1)" not in existing_data.columns:
                messagebox.showerror("Merge Error", "Cannot merge with the selected CSV file. Required columns are missing.")
                return

            # Find the next available run column index
            existing_run_columns = [col for col in existing_data.columns if col.startswith("Predicted Price (run ")]
            
            next_run_index = 1
            
            # Find the next available run column index
            while f"Predicted Price (run {next_run_index})" in existing_run_columns:
                next_run_index += 1

            # Add new run column to existing data
            new_run_column = f"Predicted Price (run {next_run_index})"
            existing_data[new_run_column] = self.analysis_information["Predicted Price"].values

            # Export merged data to the existing CSV file
            existing_data.to_csv(open_path, index=False)
            messagebox.showinfo("Merge Successful", f"Data merged and exported to {open_path} successfully!")

        # Create export buttons
        new_csv_button = tk.Button(export_window, text="Create New CSV", width=20, command=create_new_csv)
        new_csv_button.pack(padx=35, pady=15)

        merge_csv_button = tk.Button(export_window, text="Merge with Existing CSV", width=20, command=merge_with_existing_csv)
        merge_csv_button.pack(padx=35, pady=15)
        
    def _graph_stock_data(self):        
        dataset = pandas.read_csv(self.stock_data_path)
        
        # Convert the "Date" column to datetime type
        dataset['Date'] = pandas.to_datetime(dataset['Date'])
    
        # Visualize the opening price history.
        plt.title('Open Price History', fontsize=18)
        plt.plot(dataset['Date'], dataset['Open'])
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Open Price USD ($)', fontsize=16)
        plt.legend(['Current Stock'], loc='best')
        
        # Maximize the plot window
        plt.get_current_fig_manager().window.state('zoomed')
        
        plt.show()
        
    def _graph_result_data(self):
        Model.showGraph(self.dataset, self.train, self.valid, self.training_data_len)
    
    #public function to run the program
    def run(self):
        self._create_main_window()
    
    
