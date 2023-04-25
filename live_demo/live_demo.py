import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

if __name__ == "__main__":
    # Path to the folder containing the test images
    test_folder = r'C:\Users\unhmguest\OneDrive - USNH\Desktop\computer_vision\live_demo\demo_data'

    # Define the image size
    image_size = (224, 224)

    label_values = {
        0:"f1040",
        1:"f941",
        2:"w9"
    }

    # Create empty lists to store the data and labels
    data = []
    file_names = []

    print("Preprocessing the images...")
    # Loop through each file in the image folder
    for filename in tqdm(os.listdir(test_folder)):
        # Check if the file is an image file
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Generate the path to the image file
            image_path = os.path.join(test_folder, filename)

            # Load the image
            image = cv2.imread(image_path)

            # Resize the image to the desired size
            image = cv2.resize(image, image_size)

            # Convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize the pixel values to the range [0, 1]
            image = image.astype(np.float32) / 255.0

            # Add the image data and label to the lists
            data.append(image)
            file_names.append(filename)

    # Convert the data and labels to numpy arrays
    data = np.array(data)

    # Load the saved model
    model = load_model('../scripts/document_classification.h5')
    y_pred = model.predict(data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    for filename, label in zip(file_names, y_pred_labels):
        print(filename, "-->",label_values[label])