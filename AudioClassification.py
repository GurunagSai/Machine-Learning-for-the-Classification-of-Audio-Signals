import os
import glob
import librosa
import librosa.feature
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy.stats import kurtosis,skew,mode
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#Function to extract the training data and convert to numpy files
def get_traindata(path_to_train):
    
    # Load the training data csv file into a dataframe. 
    df = pd.read_csv(os.path.join(path_to_train,'TrainDataSetCSV.csv'))

    # Creating folder to store the Numpy arrays if they don't exist.
    if not os.path.exists(os.path.join(path_to_train,'train_extracted')):
        os.makedirs(os.path.join(path_to_train,'train_extracted'))

    # Getting the file names of audios from the dataframe.
    audio_files = np.array(df['file name'])
    audio_files_path = np.array(df['Relative file path'])
    # Load each audio file, save it as a numpy array
    for i in range(len(audio_files)):    
        d,r = librosa.load(os.path.join(path_to_train +'/DataSet/Training Dataset',str(audio_files_path[i]).zfill(8)))
        np.save(os.path.join(path_to_train, 'train_extracted',str(audio_files[i].replace(".wav",""))+'.npy'),d)

#Function to get test data and convert it into numpy files
def get_testdata(path_to_test):
    
    # Load the training data csv file into a dataframe. 
    df = pd.read_csv(os.path.join(path_to_test,'TestDataSetCSV.csv'))

    # Creating folder to store the Numpy arrays if they don't exist.
    if not os.path.exists(os.path.join(path_to_test,'test_extracted')):
        os.makedirs(os.path.join(path_to_test,'test_extracted'))

    # Getting the file names of audios from the dataframe.
    audio_files = np.array(df['file name'])
    audio_files_path = np.array(df['Relative file path'])
    # Load each audio file, save it as a numpy array
    for i in range(len(audio_files)):    
        d,r = librosa.load(os.path.join(path_to_test +'/DataSet/Test Dataset',str(audio_files_path[i]).zfill(8)))
        np.save(os.path.join(path_to_test, 'test_extracted',str(audio_files[i].replace(".wav",""))+'.npy'),d)

#Function to extract all the MFCC features from audio data
def get_mfcc_features(path_to_train, csv_file, extracted_folder):

  # Load the csv file into a dataframe.
  df = pd.read_csv(os.path.join(path_to_train, csv_file ))
  # Get the audio file names.
  audio_extracted = np.array(df['file name'])
  # Create an empty list to store the features.
  mfcc_features=list()
  # Looping on each Audio sequence array.
  for i in range(len(audio_extracted)):       
    # Load the Audio sequence.
    audio_file_data= np.load(os.path.join(path_to_train, extracted_folder, str(audio_extracted[i].replace(".wav",".npy"))))
    # Calculate MFCC coefficients for the audio sequence.
    mfcc_data = librosa.feature.mfcc(y=audio_file_data,sr=44100)
    # Calculating various statistic measures on the coefficients.
    mean_mfcc = np.mean(mfcc_data, axis=1)
    median_mfcc= np.median(mfcc_data,axis=1)
    std_mfcc = np.std(mfcc_data, axis=1)
    skew_mfcc = skew(mfcc_data, axis=1)
    kurt_mfcc = kurtosis(mfcc_data, axis=1)
    maximum_mfcc = np.amax(mfcc_data, axis=1)
    minimum_mfcc = np.amin(mfcc_data, axis=1)
    # Concatinating all the statistic measures and adding to the feature list.
    addList = np.concatenate((mean_mfcc,median_mfcc,std_mfcc,skew_mfcc,kurt_mfcc,maximum_mfcc,minimum_mfcc))
    mfcc_features.append(addList) 
    
  # Return feature list.
  return mfcc_features

# Function to retrive all the labels of training data
def get_labels(original_path,csv_file):

  # Load the csv file into a dataframe.
  df = pd.read_csv(os.path.join(original_path, csv_file ))

  # Return the labels.
  labels = np.array(df['Class ID'])

  return labels

#Funtion to standardize the features
def standardize_features(X_train,X_test):

  # Initialize standard scalar with zero mean
  sc = StandardScaler(with_mean=False)

  # Fit and transform the Training Dataset.
  X_train= sc.fit_transform(X_train)

  # Transform the testing set.
  X_test = sc.transform(X_test)
  
  return X_train,X_test

#Function to use Support Vector Machines classifier
def svm_classifier(X_train,Y_train,X_test):

  # Intialize SVM classifier with One-vs-Rest decision function.
  svm_model = svm.SVC(decision_function_shape='ovr')

  # Fit the Training Dataset.
  svm_model.fit(X_train, Y_train)

  # Predict and return labels.
  return svm_model.predict(X_test)

# MAIN
path_to_dir = 'C:/Users/Gurunag Sai/OneDrive/Desktop/project/AudioClassification'
path_traindata = '/DataSet/Training Dataset'
path_traindata = '/DataSet/Test Dataset'

#Run the below two lines of code to extract the datset and convert them to Numpy files
#get_traindata(path_to_dir)
#get_testdata(path_to_dir)

X_train = get_mfcc_features(path_to_dir,'TrainDataSetCSV.csv','train_extracted')
Y_train = get_labels(path_to_dir,'TrainDataSetCSV.csv')
X_test = get_mfcc_features(path_to_dir,'TestDataSetCSV.csv','test_extracted')
X_train,X_test = standardize_features(X_train,X_test)

y_test_svm = svm_classifier(X_train,Y_train,X_test)
Y_test = pd.read_csv(os.path.join(path_to_dir,'TestDataSetCSV.csv'))
Y_test['Class ID'] = y_test_svm.tolist()
Y_test = Y_test.rename(columns={"new_id":"id"})
Y_test.to_csv(os.path.join(path_to_dir,'predict_svm.csv'),index=False)