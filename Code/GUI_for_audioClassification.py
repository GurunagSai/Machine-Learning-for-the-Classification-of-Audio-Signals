#----------------------------------------------------------#
# Machine learning for the classification of audio signals #
#----------------------------------------------------------#
# *** Update the path to the folder where the code exists in the below MAIN method variables *** #
# Importing the audioClassification ML project
import AudioClassification as ac
# Libraries
import os
import numpy as np
import pandas as pd
import librosa
import librosa.feature
import librosa.display
import joblib
# Libraries to handle GUI
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#-------------------------------------------#
#---------SVM MODEL METHODS for GUI---------#
#-------------------------------------------#
# Function to train SVM model and generate confusion matrix code
def save_SVM_model(X_train, X_test, svm_model, save_file_path):    
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    # save training and testing MFCC features as numpy to the directory
    np.save(os.path.join(save_file_path,'X_train.npy'), X_train)
    np.save(os.path.join(save_file_path,'X_test.npy'), X_test)
    # Save the model
    save_file_path = os.path.join(save_file_path, 'audioClassification_SVMmodel.joblib')
    joblib.dump(svm_model, save_file_path)
    print("\nTrained SVM model is saved to: " + save_file_path)

def load_SVM_model(save_file_path):
    print("\nloading the saved trained model from - " + save_file_path)
    X_train = np.load(os.path.join(save_file_path,'X_train.npy'))
    X_test = np.load(os.path.join(save_file_path,'X_test.npy'))
    svm_model = joblib.load(os.path.join(save_file_path,'audioClassification_SVMmodel.joblib'))

    return  X_train, X_test, svm_model
    
def train_SVM_model(model_stats_desc):
    try: 
        # Run the below two lines of code to extract the datset and convert them to Numpy files
        ac.checkForNumpyData(path_to_dir, path_traindata, path_testdata) 
        saved_Model_Dir = os.path.join(path_to_dir, 'saved_SVM_model')
        if(os.path.exists(saved_Model_Dir)):
            X_train, X_test, svm_model = load_SVM_model(saved_Model_Dir)
        else:
            X_train = ac.get_mfcc_features(path_to_dir,'TrainDataSetCSV.csv','train_extracted')
            Y_train = ac.get_labels(path_to_dir,'TrainDataSetCSV.csv')
            X_test = ac.get_mfcc_features(path_to_dir,'TestDataSetCSV.csv','test_extracted')
            # Standardize the input features
            X_train_std,X_test_std = ac.standardize_features(X_train,X_test)
            # Using SVM classifier to classify the data and storing the result to SVM_MFCC_predict.csv file
            svm_model = ac.svm_classifier_train(X_train_std,Y_train)
            # Save the model
            save_SVM_model(X_train, X_test, svm_model, saved_Model_Dir)
    
        X_train_std,X_test_std = ac.standardize_features(X_train,X_test)
        y_test_svm = ac.svm_classifier_predict(svm_model, X_test_std)
        Y_test = pd.read_csv(os.path.join(path_to_dir,'TestDataSetCSV.csv'))
        Y_test['Predict Class ID'] = y_test_svm.tolist()
        Y_test = Y_test.rename(columns={"new_id":"id"})
        Y_test.to_csv(os.path.join(path_to_dir,'SVM_MFCC_predict.csv'),index=False)

        # Reading the Actual Class ID and Predicted Class ID from SVM model to generate confusion matrix and metrics
        df = pd.read_csv(os.path.join(path_to_dir,'SVM_MFCC_predict.csv'))
        Y_true = df['Actual Class ID']
        Y_test = df['Predict Class ID']
        metrics_text = ac.plot_confusion_matrix(Y_true,Y_test,"SVM_MFCC")
        model_stats_desc.config(text = metrics_text)
        model_stats_desc.pack(fill="x", pady=(0, 5))
        result_message = "Model Trained/Loaded Successfully and ready for Prediction"
        messagebox.showinfo("Prediction", result_message)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during Model Training/Loading: {str(e)}")
    
    return  X_train, svm_model

#-----------------# 
# Methods for GUI #
#-----------------#
# Function to display the audio file as a graph
def display_audio_graph(file_path, window):
    try:
        # Read audio file and create a simple plot
        data, sr = librosa.load(file_path) 
        time = np.arange(0, len(data)) / sr

        frame_plot = tk.Frame(window, padx=5, pady=5)
        frame_plot.grid(row=3, column=0)
    
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(time, data)
        ax.set_title('Audio File - ' + os.path.basename(file_path), fontdict={'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'Arial'})
        ax.set_xlabel('Time (s)', fontdict={'fontsize': 6, 'fontweight': 'bold', 'fontfamily': 'Times New Roman'})
        ax.set_ylabel('Amplitude', fontdict={'fontsize': 6, 'fontweight': 'normal', 'fontfamily': 'Times New Roman'})
        ax.grid()
        # Plotting the audio file in the GUI
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while displaying the audio file: {str(e)}")

# Function to perform prediction and display/save results
def perform_prediction_single(result_description_single, entry_path, save_file_path):
    input_file_path = entry_path.get()

    if not input_file_path:
        messagebox.showwarning("Warning", "Please select an audio file.")
        return

    try:
        label_names=["Black","Red","Silver"]
        X_train = np.load(os.path.join(save_file_path,'X_train.npy'))
        X_test = np.load(os.path.join(save_file_path,'X_test.npy'))
        svm_model = joblib.load(os.path.join(save_file_path,'audioClassification_SVMmodel.joblib'))
        # Extract the audio data and necessary MFCC features for prediction
        data, sr = librosa.load(input_file_path)
        mfcc_data = librosa.feature.mfcc(y=data,sr=44100)
        X_predict = ac.extract_mfcc_data(mfcc_data)
        sc = ac.StandardScaler(with_mean=False)
        # Prediction of the audio file
        X_train_std= sc.fit_transform(X_train)
        X_predict = sc.transform(X_predict.reshape(1,-1)) 
        y_predict_svm = ac.svm_classifier_predict(svm_model, X_predict)
        
        # Display prediction results in GUI label
        result_message = "Predicted Class ID:" + str(y_predict_svm[0]) + "\nPredicted Label: " + str(label_names[y_predict_svm[0]-1])
        result_description_single.config(text = result_message)
        result_description_single.pack(side=tk.BOTTOM, fill="x", pady=(0, 5))

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

# Method for GUI to predict multiple files
def perform_prediction_multiple(result_description_folder, folder_description, entry_path, save_file_path):
    input_file_path = entry_path.get()
    if not input_file_path:
        messagebox.showwarning("Warning", "Please select a folder!")
        return
    wav_files_path = []
    wav_files_filename = []
    # List all files in the directory
    for file in os.listdir(input_file_path):
        # Check if the file has a ".wav" extension
        if file.endswith(".wav"):
            wav_files_path.append(os.path.join(input_file_path, file))
            wav_files_filename.append(os.path.basename(file))
    try:
        label_names=["Black","Red","Silver"]
        X_train = np.load(os.path.join(save_file_path,'X_train.npy'))
        X_test = np.load(os.path.join(save_file_path,'X_test.npy'))
        svm_model = joblib.load(os.path.join(save_file_path,'audioClassification_SVMmodel.joblib'))
        X_predict=list()
        for audio in wav_files_path:
            data, sr = librosa.load(audio)
            mfcc_data = librosa.feature.mfcc(y=data,sr=44100)
            addList = ac.extract_mfcc_data(mfcc_data)
            X_predict.append(addList) 
        sc = ac.StandardScaler(with_mean=False)
        # Prediction of the audio file
        X_train_std= sc.fit_transform(X_train)
        X_predict = sc.transform(X_predict) 
        y_predict_svm = ac.svm_classifier_predict(svm_model, X_predict)
        y_predict_label = []
        for predict_result in y_predict_svm:
            y_predict_label.append(label_names[predict_result-1])
        # Results
        global prediction_folder_csv 
        prediction_folder_csv  = os.path.join(input_file_path, "Prediction_Results.csv")
        df = pd.DataFrame({'File Name': wav_files_filename, 'Prediction class ID': y_predict_svm, 'Prediction Label': y_predict_label})
        df.to_csv(prediction_folder_csv, index=False)
        result_description_folder.config(text = "Prediction results stored in: \n" + prediction_folder_csv)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

# Funmtion that opens the prediction results csv file
def open_csv(csv_file):
    if csv_file:
        os.system(f'start {csv_file}')
    else:
        messagebox.ERROR("Error in Prediction result CSV", "Prediction results csv file do not exist!")

# Function to display prediction csv file in a frame of GUI
def display_csv_in_GUI(frame, csv_file):
    if not(csv_file):
        messagebox.ERROR("Error in Prediction result CSV", "Prediction results csv file do not exist!")
    else:
        df = pd.read_csv(csv_file)
        # Create a scrollbar for the frame
        scrollbar = tk.Scrollbar(frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        # Create a text widget to display the CSV contents
        text_widget = tk.Text(frame, yscrollcommand=scrollbar.set)
        text_widget.pack(fill="both", expand=True)
        # Insert the CSV contents into the text widget
        text_widget.insert("1.0", df.to_string(index=False))
        # Configure the scrollbar
        scrollbar.config(command=text_widget.yview)

# Function to define GUI
def create_main_gui():
    window = tk.Tk()
    window.title("Audio Classification")

    #--- Section 1: Model Training
    # Create a frame for model training
    frame_train1 = tk.Frame(window, padx=5, pady=5)
    frame_train1.grid(row=0, column=0)
    # Create a label for the model training section
    label_train = tk.Label(frame_train1, text="Model Training", font=("Times New Roman", 14, "bold", "underline"))
    label_train.pack(fill="x", pady=5)
    # Create a sublabel for additional instructions
    sublabel_train = tk.Label(frame_train1, text="Please train the model for Prediction\nHint: Click the below 'Train Model' button to train the model", font=("Times New Roman", 10))
    sublabel_train.pack(fill="x", pady=5)
    # Create a button to trigger model training
    btn_train = tk.Button(frame_train1, text="Train Model", command=lambda: train_SVM_model(model_stats_desc))
    btn_train.pack()
    frame_train2 = tk.Frame(window, padx=5, pady=5)
    frame_train2.grid(row=0, column=2)
    # Create a label to display model statistics
    model_stats = tk.Label(frame_train2, text="Model statistics:", font=("Times New Roman", 11))
    model_stats.pack(fill="x", pady=5)
    # Create a label to display detailed model statistics
    model_stats_desc = tk.Label(frame_train2, text="Model stats:", font=("Times New Roman", 9))
    
    # Add a separator between Model Training and Single File Prediction
    ttk.Separator(window, orient='horizontal').grid(row=1, columnspan=3, sticky='ew')
    
    #--- Section 2: Single File Prediction
    # Create a frame for the single audio file prediction section
    frame_single_prediction1 = tk.Frame(window, padx=5, pady=5)
    frame_single_prediction1.grid(row=2, column=0)
    # Create a label for the single audio file prediction section
    label_single_prediction_heading = tk.Label(frame_single_prediction1, text="Single Audio File Prediction", font=("Times New Roman", 14, "bold", "underline"))
    label_single_prediction_heading.pack(fill="x", pady=2)
    # Description text for the the single audio file prediction section
    label_single_prediction_description = tk.Label(frame_single_prediction1, text="Please select an audio file for class prediction\nHint: Click 'Browse', select an audio file, and click 'Predict' button", font=("Times New Roman", 10))
    label_single_prediction_description.pack(fill="x", pady=2)
    # File path entry for the the single audio file prediction section
    entry_single_path = tk.Entry(frame_single_prediction1, width=50)
    entry_single_path.pack(fill="x", pady=2, side=tk.TOP)
    # Function to browse for a WAV file and display it as a graph
    def browse_file_wav():
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        entry_single_path.delete(0, tk.END)
        entry_single_path.insert(0, file_path)
        result_description_single.config(text = "")
        # Clear the canvas and display the selected audio file as a graph
        display_audio_graph(file_path, window)
    # Browse button to select file path
    btn_browse = tk.Button(frame_single_prediction1, text="Browse", command=browse_file_wav)
    btn_browse.pack(pady=5, padx=5, side=tk.TOP)   
    # Predict button
    btn_predict_single = tk.Button(frame_single_prediction1, text="Predict", command=lambda: perform_prediction_single(result_description_single, entry_single_path, os.path.join(path_to_dir, 'saved_SVM_model')))
    btn_predict_single.pack(side=tk.TOP, pady=5, padx=5)
    # Label for prediction result description
    result_description_single = tk.Label(frame_single_prediction1, text = "Predicted Class ID:\nPredicted Label: ", font=("Times New Roman", 9,  "bold"))
    result_description_single.pack(side=tk.TOP, pady=5, padx=5)
    
    # Add a separator between Prediction Single File and Prediction Multiple Files
    ttk.Separator(window, orient='vertical').grid(column=1, row=2, rowspan=3, sticky='ns')
    
    #--- Section 3: Folder Prediction
    # Create a frame for folder prediction
    frame_folder_prediction1 = tk.Frame(window, padx=5, pady=5)
    frame_folder_prediction1.grid(row=2, column=2)
    # Create a label for the folder prediction section
    label_folder_prediction_heading = tk.Label(frame_folder_prediction1, text="Multiple Audio File Prediction", font=("Times New Roman", 14, "bold", "underline"))
    label_folder_prediction_heading.pack(fill="x", pady=2)
    # Description text for the the single audio file prediction section
    label_folder_prediction_description = tk.Label(frame_folder_prediction1, text="Please select a folder which contains the audio files for prediction\nHint: Click 'Browse', Select folder which contains audio files,\nand click 'predict' button", font=("Times New Roman", 10))
    label_folder_prediction_description.pack(fill="x", pady=2)
    # Create an entry widget for entering the folder path
    entry_folder_path = tk.Entry(frame_folder_prediction1, width=50)
    entry_folder_path.pack(fill="x", pady=2, side=tk.TOP)
    # Function to browse for a WAV file and display it as a graph
    def browse_file_folder():
        folder_path = filedialog.askdirectory()
        entry_folder_path.delete(0, tk.END)
        entry_folder_path.insert(0, folder_path)
        result_description_folder.config(text = "")
        if folder_path:
            print("Selected folder:", folder_path)
            count = 0
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    count += 1
            folder_description.config(text = "Prediction folder has " + str(count) + " audio files,\n click on PREDICT for audio data prediction...")
    # Browse button to select file path
    btn_browse = tk.Button(frame_folder_prediction1, text="Browse", command=browse_file_folder)
    btn_browse.pack(pady=5, padx=5, side=tk.TOP)
    # Create a button to trigger prediction for the entire folder
    btn_predict_folder = tk.Button(frame_folder_prediction1, text="Predict Files", command=lambda: perform_prediction_multiple(result_description_folder, folder_description, entry_folder_path, os.path.join(path_to_dir, 'saved_SVM_model')))
    btn_predict_folder.pack(padx=5, pady=5, side=tk.TOP)
    folder_description = tk.Label(frame_folder_prediction1, text = ".wav files in the folder:", font=("Times New Roman", 9,  "bold"))
    folder_description.pack(side=tk.TOP, pady=5, padx=5)
    # Result decription folder label and button to open
    frame_folder_prediction2 = tk.Frame(window, padx=5, pady=5)
    frame_folder_prediction2.grid(row=3, column=2, sticky="nsew")
    result_description_folder = tk.Label(frame_folder_prediction2, text = "Prediction results stored in:", font=("Times New Roman", 9,  "bold"))
    result_description_folder.pack(side=tk.TOP, pady=5, padx=5)
    btn_open_csv = tk.Button(frame_folder_prediction2, text="Open CSV File", command=lambda: open_csv(prediction_folder_csv))
    btn_open_csv.pack(side=tk.TOP, pady=5, padx=5)
    # Display csv in GUI
    btn_display_csv = tk.Button(frame_folder_prediction2, text="Display CSV File", command=lambda: display_csv_in_GUI(frame_folder_prediction2, prediction_folder_csv))
    btn_display_csv.pack(side=tk.TOP, pady=5, padx=5)
    
    return window
    
#-------------------------------------------#
#-------------------MAIN--------------------#
#-------------------------------------------#
# Update the path to the folder where code exists
path_to_dir = 'D:/FUAS/Subjects/sem-3/project/AudioClassification'   
path_traindata = 'DataSet/Training Dataset'
path_testdata = 'DataSet/Test Dataset'

# Create the main GUI
main_window = create_main_gui()

# Run the GUI
main_window.mainloop()
