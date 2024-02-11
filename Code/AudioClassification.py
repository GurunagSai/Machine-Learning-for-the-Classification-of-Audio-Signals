#----------------------------------------------------------#
# Machine learning for the classification of audio signals #
#----------------------------------------------------------#
import os
import librosa
import librosa.feature
import librosa.display
import numpy as np
import pandas as pd
from scipy.stats import kurtosis,skew
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Function to extract the training data and convert to numpy files
def get_traindata(dir_path,path_traindata):
    # Load the training data csv file into a dataframe. 
    df = pd.read_csv(os.path.join(dir_path,'TrainDataSetCSV.csv'))
    
    # Creating folder to store the Numpy arrays if they don't exist.
    if not os.path.exists(os.path.join(dir_path,'train_extracted')):
        os.makedirs(os.path.join(dir_path,'train_extracted'))
        
    # Getting the file names of audios from the dataframe.
    audio_files = np.array(df['file name'])
    audio_files_path = np.array(df['Relative file path'])
    
    # Load each audio file, save it as a numpy array
    for i in range(len(audio_files)):   
        path = os.path.join(dir_path, path_traindata, str(audio_files_path[i]).zfill(8))
        d,r = librosa.load(path)
        np.save(os.path.join(dir_path, 'train_extracted',str(audio_files[i].replace(".wav",""))+'.npy'),d)

# Function to get test data and convert it into numpy files
def get_testdata(dir_path, path_testdata):
    
    # Load the training data csv file into a datwaframe. 
    df = pd.read_csv(os.path.join(dir_path,'TestDataSetCSV.csv'))
    
    # Creating folder to store the Numpy arrays if they don't exist.
    if not os.path.exists(os.path.join(dir_path,'test_extracted')):
        os.makedirs(os.path.join(dir_path,'test_extracted'))
        
    # Getting the file names of audios from the dataframe.
    audio_files = np.array(df['file name'])
    audio_files_path = np.array(df['Relative file path'])
    
    # Load each audio file, save it as a numpy array
    for i in range(len(audio_files)): 
        path=os.path.join(dir_path, path_testdata, str(audio_files_path[i]))
        d,r = librosa.load(path)
        np.save(os.path.join(dir_path,'test_extracted',str(audio_files[i].replace(".wav",""))+'.npy'),d)
        
# Function to check if the training and test data exists in numpy format
def checkForNumpyData(path_to_dir, path_traindata, path_testdata):
    if(not(os.path.exists("train_extracted"))):
        get_traindata(path_to_dir,path_traindata)
    elif(not(os.path.exists("test_extracted"))):
        get_testdata(path_to_dir,path_testdata)
    else:
        print("The training data and test data exists as numpy")

#-----------------------------------------#
#----------SVM with MFCC METHODS----------#
#-----------------------------------------#
# Function to extract all the MFCC features from audio data
def get_mfcc_features(path_to_train, csv_file, extracted_folder):
    # Load the csv file into a dataframe.
    df = pd.read_csv(os.path.join(path_to_train, csv_file ))
    audio_extracted = np.array(df['file name'])
    mfcc_features=list()
  
    # Looping on each Audio sequence array.
    for i in range(len(audio_extracted)):  
        # Load the Audio sequence.
        audio_file_data= np.load(os.path.join(path_to_train, extracted_folder, str(audio_extracted[i].replace(".wav",".npy"))))
        # Calculate MFCC coefficients for the audio sequence.
        mfcc_data = librosa.feature.mfcc(y=audio_file_data,sr=44100)
        # Concatinating all the statistic measures and adding to the feature list.
        addList = extract_mfcc_data(mfcc_data)
        mfcc_features.append(addList) 
    
    # Return feature list.
    return mfcc_features

def extract_mfcc_data(mfcc_data):
	# Calculating various statistic measures on the coefficients.
    mean_mfcc = np.mean(mfcc_data, axis=1)
    median_mfcc= np.median(mfcc_data,axis=1)
    std_mfcc = np.std(mfcc_data, axis=1)
    skew_mfcc = skew(mfcc_data, axis=1)
    kurt_mfcc = kurtosis(mfcc_data, axis=1)
    maximum_mfcc = np.amax(mfcc_data, axis=1)
    minimum_mfcc = np.amin(mfcc_data, axis=1)
    # Concatinating all the statistic measures and adding to the feature list.
    return	np.concatenate((mean_mfcc,median_mfcc,std_mfcc,skew_mfcc,kurt_mfcc,maximum_mfcc,minimum_mfcc))

# Function to retrive all the labels of training data
def get_labels(original_path,csv_file):
    # Load the csv file into a dataframe.
    df = pd.read_csv(os.path.join(original_path, csv_file ))
    # Return the labels.
    labels = np.array(df['Class ID'])

    return labels

# Funtion to standardize the features
def standardize_features(X_train,X_test):
    # Initialize standard scalar with zero mean
    sc = StandardScaler(with_mean=False)
    # Fit and transform the Training Dataset.
    X_train= sc.fit_transform(X_train)
    # Transform the testing set.
    X_test = sc.transform(X_test)
  
    return X_train,X_test

# Function to create Support Vector Machines classifier and train
def svm_classifier_train(X_train,Y_train):
    # Intialize SVM classifier with One-vs-Rest decision function.
    svm_model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    # Fit the Training Dataset.
    svm_model.fit(X_train, Y_train)
  
    return svm_model

# Function to use SVM model for prediction
def svm_classifier_predict(svm_model, X_test):
  # Predict labels for the test dataset.
  Y_test = svm_model.predict(X_test)
  
  return Y_test

#-----------------------------------------#
#---------CNN SPECTROGRAM METHODS---------#
#-----------------------------------------#
#Function to preprocess the audio data with spectrogram
def get_spectrograms(path_to_dir, train_csv , train_extracted, test_csv ,test_extracted, target_shape=(128, 128)):
    # Read the train_csv for Training Dataset file names.
    df = pd.read_csv(os.path.join(path_to_dir, train_csv ))
    audio_extracted = np.array(df['file name'])
    labels = np.array(df['Class ID'])

    # Intializing empty list for storing spectrograms and labels.
    X_train = []
    Y_train = []
    X_test = []  

    # Function to resize or pad the spectrogram to the target shape
    def resize_spectrogram(spec, target_shape):
        if spec.shape != target_shape:
            spec = np.pad(spec, ((0, max(0, target_shape[0] - spec.shape[0])), (0, max(0, target_shape[1] - spec.shape[1]))), mode='constant')
        return spec

     # Looping through the Training Data Audio sequences.
    for i in range(len(audio_extracted)):
        # Load Audio Sequence, calculate the Mel spectrogram and append it to list.
        audio_file_data = np.load(os.path.join(path_to_dir, train_extracted, str(audio_extracted[i].replace(".wav", ".npy"))))
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=audio_file_data, n_mels=target_shape[0], hop_length=2048), ref=np.max)
        mel_spec_resized = resize_spectrogram(mel_spec, target_shape)
        X_train.append(mel_spec_resized)
        Y_train.append(labels[i])  

    # Read the test_csv for Testing Dataset file names.
    df = pd.read_csv(os.path.join(path_to_dir, test_csv))
    audio_extracted = np.array(df['file name'])

    # Looping through the testing data audio sequences.
    for i in range(len(audio_extracted)):
        # Load Audio Sequence, calculate the Mel spectrogram and append it to list.
        audio_file_data = np.load(os.path.join(path_to_dir, test_extracted, str(audio_extracted[i].replace(".wav", ".npy"))))
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=audio_file_data, n_mels=target_shape[0], hop_length=2048), ref=np.max)
        mel_spec_resized = resize_spectrogram(mel_spec, target_shape)
        X_test.append(mel_spec_resized)
  
  
    # Converting the lists into numpy arrays.
    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape(len(Y_train),1)
    X_test = np.array(X_test)

    # Splitting the training features into Training and Validation.
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratified_split.split(X_train, Y_train):
        training_data,train_labels=X_train[train_index],Y_train[train_index]
        val_data,val_labels=X_train[test_index],Y_train[test_index] 
  
    # One hot encoding the Training and validation labels.
    Y_train = to_categorical(train_labels,4)
    Y_val = to_categorical(val_labels,4)

    # Reshaping the features into the form (rows, columns, 1) for Convolutional Neural Networks.
    X_train = [training_data[i].reshape(target_shape[0], target_shape[1], 1) for i in range(len(training_data))]
    X_val = [val_data[i].reshape(target_shape[0], target_shape[1], 1) for i in range(len(val_data))]
    X_test = [X_test[i].reshape(target_shape[0], target_shape[1], 1) for i in range(len(X_test))]

    return  X_train,Y_train,X_val,Y_val,X_test

# Function to use CNN classifier
def cnn_classifier(inputShape):
    # Intializing the model sequential.
    model = Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(16, (4, 4), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(32, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

    print(model.summary())
    
    return model
  
#-----------------------------------------#
#---------PLOT CONFUSIUON MATRIX----------#
#-----------------------------------------#
# Function to plot the confusion matrix
def plot_confusion_matrix(y_true,y_pred,classifier):
    
    label_names=["Black","Red","Silver"]
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Calculate the confusion matrix and metrics using the expected and predicted values.
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100

    # Calculate additional metrics
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    # Convert percentages
    TPR_percentage = np.round(TPR * 100, 2)
    TNR_percentage = np.round(TNR * 100, 2)
    FPR_percentage = np.round(FPR * 100, 2)
    FNR_percentage = np.round(FNR * 100, 2)
    
    # Show the confusion matrix values.
    fig = plt.figure(figsize=(5,8))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')  

    # Display metrics at the bottom of the plot
    metrics_text = (
        f'Accuracy: {np.round(accuracy, 2)}%  '
        f'Precision: {np.round(precision, 2)}%\n'
        f'Recall: {np.round(recall, 2)}%  '
        f'F1 Score: {np.round(f1, 2)}%\n'
        f"TPR in %: {TPR_percentage}\n"
        f"TNR in %: {TNR_percentage}\n"
        f"FPR in %: {FPR_percentage}\n"
        f"FNR in %: {FNR_percentage}"
    ) 
    
    # Set the x, y and title labels for the plot.
    plt.xlabel("Expected Outputs", fontsize=10)
    plt.ylabel("Actual Outputs", fontsize=10)
    plt.title("Confusion Matrix of "+ classifier + " classifier",fontsize=12)  
    
    # Arrange the label names on the x and y axis.
    plt.xticks(np.arange(len(label_names)), label_names, rotation='horizontal')
    plt.yticks(np.arange(len(label_names)), label_names,)
    plt.tick_params(axis='both', labelsize='10')
    plt.tight_layout()
    for (y, x), label in np.ndenumerate(conf_matrix):
        if label != 0:
            plt.text(x,y,label,ha='center',va='center', size='12')
                 
    # Display metrics on the plot
    plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12)
    
    # Save and Show the plot
    plt.savefig(classifier + "_confusion_matrix.png")
    plt.show()
    
    return	metrics_text

#--------------------------------------------------#
#---------CNN with SPECTROGRAM MODEL MAIN----------#
#--------------------------------------------------#
""" # Get the spectrograms
X_train,Y_train,X_val,Y_val, X_test = get_spectrograms(path_to_dir,'TrainDataSetCSV.csv','train_extracted', 'TestDataSetCSV.csv','test_extracted')

# Applying 2-D Convolution Neural Network on the spectrograms.
model = cnn_classifier(X_train[0].shape)
model.fit(np.array(X_train), Y_train, epochs=20, batch_size=16, validation_data=(np.array(X_val), Y_val))

# Predict the Test dataset using the model and save results to a CSV file.
Y_test_nn = np.argmax(model.predict(np.array(X_test)),axis=1)
Y_test = pd.read_csv(os.path.join(path_to_dir,'TestDataSetCSV.csv'))
Y_test['Predict Class ID'] = Y_test_nn.tolist()
Y_test = Y_test.rename(columns={'new_id':'id'})
Y_test.to_csv(os.path.join(path_to_dir,'CNN_Spectrogram.csv'),index=False)

# Plot Confusion matrix using validation predictions.
df = pd.read_csv(os.path.join(path_to_dir,'CNN_Spectrogram.csv'))
Y_true = df['Actual Class ID']
Y_test = df['Predict Class ID']
plot_confusion_matrix(Y_true,Y_test,"CNN_Spectrogram") """
