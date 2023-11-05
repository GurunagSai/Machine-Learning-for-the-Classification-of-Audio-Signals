import os
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy.stats import kurtosis,skew,mode

#Function to extract the training data
def get_training(path_to_train):
    
    # Load the training data csv file into a dataframe. 
    df = pd.read_csv(os.path.join(path_to_train,'DataSetCSV.csv'))

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
    audio_file_data= np.load(os.path.join(path_to_train, extracted_folder, str(audio_extracted[i])+'.npy'))

    # Calculate MFCC coefficients for the audio sequence.
    mfcc_data = librosa.feature.mfcc(y=audio_file_data,sr=22050)

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

# MAIN
path_to_dir = 'C:/Users/Gurunag Sai/OneDrive/Desktop/project/AudioClassification'
path_trainingdata = '/DataSet/Training Dataset'
#get_training(path_to_dir)
X_train = get_mfcc_features(path_to_dir,'DataSetCSV.csv','train_extracted')