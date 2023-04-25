import cv2
import os
import pandas as pd
from tqdm import tqdm

# Path to the folder containing the images
image_folder = '../data/'

# Create an empty list to store the data
data_list = []

# Loop through each file in the image folder
for filename in tqdm(os.listdir(image_folder)):
    # Check if the file is an image file
    if filename.endswith('.png'):
        # Generate the path to the image file
        image_path = os.path.join(image_folder, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Get the filename and the first slice of it
        filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        result = filename_without_ext.split('_')[0]

        # Add the data to the data list
        data_list.append({'filename': filename_without_ext, 'result': result})

# Create a pandas DataFrame with the data list
df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
df.to_csv('../pdf/results.csv', index=False)
