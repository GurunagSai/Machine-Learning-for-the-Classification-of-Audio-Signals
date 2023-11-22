import os
import csv
import re
import librosa

# Training data set directory
directory_path = 'C:/Users/Gurunag Sai/OneDrive/Desktop/project/AudioClassification/DataSet/Test DataSet/'

srno_append = -700

# Initialize a list to store the file information
file_info = []

# Iterate through the files in the training directory
for root, dirs, files in os.walk(directory_path):
    print(root)
    print(dirs)
    print(srno_append)
    for filename in files:
        
        # Get all the info relkated to file name and path and add to respective file property
        file_path = os.path.join(root, filename)
        file_name = file_path.replace(directory_path,"")
        
        serial_number = file_name.replace('test',"")
        serial_number = int(serial_number.replace('.wav', ""))
        # audio data
        audio_data, sampling_rate = librosa.load(os.path.join(root, filename), sr=None)
        audio_duration = librosa.get_duration(y=audio_data, sr=sampling_rate)

        # Append the information to the file_info list
        file_info.append([serial_number, file_name, file_path, sampling_rate, audio_duration])
    if dir:
        srno_append = srno_append + 700

# Define the CSV file name
csv_file = 'TestDataSetCSV.csv'

# Write the information to a CSV file
with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Header of CSV file
    csv_writer.writerow(['Serial Number', 'file name', 'Relative file path', 'Sampling rate(Hz)', 'Audio Duration(sec)'])
    # Writing the data
    csv_writer.writerows(file_info)

