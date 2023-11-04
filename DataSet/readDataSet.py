import os
import csv
import re

# Training data set directory
directory_path = 'C:/Users/Gurunag Sai/OneDrive/Desktop/project/AudioClassification/DataSet/Training Dataset/'

srno_append = 0

# Initialize a list to store the file information
file_info = []

# Iterate through the files in the training directory
for root, dirs, files in os.walk(directory_path):
    for filename in files:
        
        # Get all the info relkated to file name and path and add to respective file property
        file_path = os.path.join(root, filename)
        file_path = file_path.replace(directory_path,"")
        
        class_name = os.path.basename(os.path.dirname(file_path))
        
        num = [int(x) for x in file_path.split() if x.isdigit()] 
        serial_number = num + srno_append
        
        # Naming class_id as per the file name
        if class_name == 'black':
            class_id = 1
        elif class_name == 'red':
            class_id = 2
        elif class_name == 'silver':
            class_id = 3
        else:
            class_id = 4

        # Append the information to the file_info list
        file_info.append([serial_number, class_name, class_id, file_path])
    srno_append = 4500

# Define the CSV file name
csv_file = 'DataSetCSV.csv'

# Write the information to a CSV file
with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Header of CSV file
    csv_writer.writerow(['Serial Number', 'Class Name', 'Class ID', 'Relative file path'])
    # Writing the data
    csv_writer.writerows(file_info)

