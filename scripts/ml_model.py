import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

label_values = {
    "f1040":0,
    "f941":1,
    "w9":2
}

if __name__ == "__main__":

    # Path to the folder containing the images
    image_folder = r'C:\Users\unhmguest\OneDrive - USNH\Desktop\computer_vision\rendered_images'

    # Define the image size
    image_size = (224, 224)

    # Create empty lists to store the data and labels
    data = []
    labels = []

    print("Preprocessing the images...")
    # Loop through each file in the image folder
    for filename in tqdm(os.listdir(image_folder)):
        # Check if the file is an image file
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Generate the path to the image file
            image_path = os.path.join(image_folder, filename)

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
    # print(labels)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert the labels to one-hot encoded vectors
    num_classes = len(np.unique(labels))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(y_train)

    # # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Define the checkpoint to save the best model
    checkpoint = ModelCheckpoint('document_classification.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # Train the model
    batch_size = 128
    epochs = 10
    steps_per_epoch = X_train.shape[0] // batch_size
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[checkpoint])
    model = load_model('document_classification.h5')
    print("Job Complete")