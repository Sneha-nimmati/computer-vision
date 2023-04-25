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
    test_folder = r'C:\Users\unhmguest\OneDrive - USNH\Desktop\computer_vision\test_images'

    # Define the image size
    image_size = (224, 224)

    # Define the data generator for normalization
    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    label_values = {
        "f1040":0,
        "f941":1,
        "w9":2
    }

    # Create empty lists to store the data and labels
    data = []
    labels = []

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
            label = filename.split('_')[0]
            label_value = label_values[label]
            labels.append(label_value)

    # Convert the data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Load the saved model
    model = load_model('document_classification.h5')
    y_pred = model.predict(data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print(y_pred_labels)
    print(labels)
    print(classification_report(labels, y_pred_labels, target_names=label_values.keys()))
    print(confusion_matrix(labels, y_pred_labels))