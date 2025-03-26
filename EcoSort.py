import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib.pyplot import imread, imshow
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def VGG_Waste_Classifier(input_tensor=None, classes=2):    
    img_dim = (300, 300, 3)
    img_input = Input(shape=img_dim)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x, name='WasteClassifier')
    return model

waste_model = VGG_Waste_Classifier(classes=2)
waste_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_and_preprocess_waste_data(data_directory, img_size=300):
    data, labels = [], []
    waste_types = os.listdir(data_directory)
    for waste_type in waste_types:
        waste_type_path = os.path.join(data_directory, waste_type)
        if not os.path.isdir(waste_type_path):
            continue
        for class_type in os.listdir(waste_type_path):
            class_path = os.path.join(waste_type_path, class_type)
            if not os.path.isdir(class_path):
                continue
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if not os.path.isfile(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(class_type)
    return np.array(data), np.array(labels)

dataset_directory = r'C:\Users\abhra\Downloads\data_directory'
data, labels = load_and_preprocess_waste_data(dataset_directory)

waste_categories = ['bio-degradable', 'non-bio-degradable']
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded, num_classes=2)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
train_data = train_data.reshape((-1, 300, 300, 3))
test_data = test_data.reshape((-1, 300, 300, 3))

history = waste_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

loss, accuracy = waste_model.evaluate(test_data, test_labels, batch_size=32)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

img_path = input("Enter the path to the image: ")
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(300, 300))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

imshow(imread(img_path))
plt.show()

prediction = waste_model.predict(x)
predicted_category = waste_categories[np.argmax(prediction)]
print("Prediction:", predicted_category)
