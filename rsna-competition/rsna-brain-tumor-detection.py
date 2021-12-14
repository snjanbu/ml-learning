import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
import pydicom
import csv
import gc
from tqdm import tqdm
import tensorflow as tf
import albumentations
import time


class CustomSequentialModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.layer1 = Conv2D(32, input_shape=(64, 64, 1), activation=activation, kernel_size=(3, 3))
        self.layer2 = BatchNormalization()
        self.layer3 = Flatten()
        self.layer4 = Dense(32, activation=activation, kernel_initializer='he_normal')
        self.layer5 = Dropout(0.15)
        self.model_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform')

    def call(self, model_input):
        op_layer = self.layer1(tf.dtypes.cast(model_input, tf.float32))
        op_layer = self.layer2(op_layer)
        op_layer = self.layer3(op_layer)
        op_layer = self.layer4(op_layer)
        op_layer = self.layer5(op_layer)
        model_output = self.model_output(op_layer)
        return model_output


base_folder = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_data = pd.read_csv(os.path.join(base_folder, "train_labels.csv"))
excluded_patients = ["00109", "00123", "00709"]
image_types = ["T1wCE", "FLAIR", "T1w", "T2w"]
image_types = ["T1w"]


def construct_image_df(test_type):
    image_list = []
    for patient_id in os.listdir(os.path.join(base_folder, test_type)):
        if patient_id not in excluded_patients:
            patient_record_value = train_data[train_data["BraTS21ID"] == int(patient_id)]['MGMT_value'].item()
            for image_type in image_types:
                folder_dir = os.path.join(base_folder, test_type, patient_id, image_type)
                for file_name in os.listdir(folder_dir):
                    image_list.append({"file_path": os.path.join(folder_dir, file_name), "value": str(patient_record_value), "patient_id":patient_id, "image_type": image_type})
    return pd.DataFrame(image_list)

image_df = construct_image_df("train")
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=7)
test_df.head()


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, batch_size, df):
        self.batch_size = batch_size
        self.shuffle = True
        self.df = df
        self.n = len(self.df)

    def __len__(self):
        l = int(len(self.df) / self.batch_size)
        if l * self.batch_size < len(self.df):
            l += 1
        return l

    def __get_resized_image(self, image):
        image_arr = cv2.resize(image.pixel_array, (64, 64))

        return image_arr

    def __get_output(self, label, classes):
        return to_categorical(label, num_classes=classes)

    def __get_cropped_image(self, image):
        cropped_image = self.crop_pipeline(image=image.pixel_array)["image"]
        return cv2.resize(cropped_image, (64, 64))

    def __get_data(self, batches):
        X_batch, y_batch = [], []
        for index, row in batches.iterrows():
            image = pydicom.read_file(row['file_path'])
            if (np.amax(image.pixel_array) != 0):
                X_batch.append(self.__get_resized_image(image))
                y_batch.append(self.__get_output(row['value'], 2))
        return np.expand_dims(X_batch, axis=-1), np.array(y_batch)

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y


train_datagen = CustomDataGen(batch_size=512, df=train_df)
valid_datagen = CustomDataGen(batch_size=512, df=test_df)



# model = CustomSequentialModel()
model = Sequential()

model.add(Conv2D(16, input_shape=(64, 64, 1), activation='relu', kernel_size=(4,4)))
model.add(BatchNormalization())
model.add(MaxPooling2D(4,4))

model.add(Conv2D(16, activation='relu', kernel_size=(4,4)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(16, activation='relu', kernel_size=(4,4)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(16, activation='relu', kernel_size=(1, 1)))
model.add(Dense(8, activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))

model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[AUC()])
# model.build(input_shape=(None, 64, 64, 1))
model.summary()


history = model.fit(train_datagen,validation_data=valid_datagen, epochs=10, steps_per_epoch=len(train_df)/ 512)

def construct_test_df(test_type="test"):
    image_list = []
    for patient_id in os.listdir(os.path.join(base_folder, test_type)):
        if patient_id not in excluded_patients:
            for image_type in image_types:
                folder_dir = os.path.join(base_folder, test_type, patient_id, image_type)
                for file_name in os.listdir(folder_dir):
                    image_list.append({"file_path": os.path.join(folder_dir, file_name), "patient_id":patient_id, "image_type": image_type})
    return pd.DataFrame(image_list)

test_df = construct_test_df()
test_df.head()

test_df.groupby(['image_type']).agg(['count'])


class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size

    def __get_input_data(self, batches):
        img_arr = []
        for index, row in batches.iterrows():
            image = pydicom.read_file(row['file_path'])
            img_arr.append(cv2.resize(image.pixel_array, (64, 64)))
        img_arr = np.expand_dims(img_arr, axis=-1)
        return img_arr

    def __len__(self):
        l = int(len(self.df) / self.batch_size)
        if l * self.batch_size < len(self.df):
            l += 1
        return l

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        X = self.__get_input_data(batches)
        return X


test_datagen = TestDataGenerator(batch_size=256, df=test_df)
output = model.predict(test_datagen)


modified_output = np.amax(output, axis=1)

test_df['MGMT_value'] = modified_output

test_df


result_df = test_df.groupby('patient_id', as_index=False).agg({"MGMT_value": ['mean']}).reset_index()
result_df.columns = ['id', 'BraTS21ID', 'MGMT_value']
result_df['BraTS21ID'] = result_df['BraTS21ID'].astype('string')


result_df[["BraTS21ID", "MGMT_value"]].head()


result_df[["BraTS21ID", "MGMT_value"]].tail()
mod_result_df = result_df.copy()


mod_result_df['MGMT_value'] = mod_result_df['MGMT_value'].round(1)

mod_result_df[['BraTS21ID', 'MGMT_value']].to_csv('submission.csv', index=False)

mod_result_df.shape


mod_result_df.head()