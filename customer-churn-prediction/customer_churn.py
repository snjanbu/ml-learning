import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("Telco-Customer-Churn.csv")

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

def find_unique_values(df):
    for column in df:
        if pd.api.types.is_string_dtype(df[column]) != True:
            print(f'{column} : {df[column].unique()}')


find_unique_values(data)

replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender', 'InternetService', "Contract", "PaymentMethod"]
for col in replace_cols:
    data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    data[col] = data[col].replace({"Yes": 1, "No": 0, "Female": 0, "Male": 1, "DSL":1, "Fiber optic": 2, "Month-to-month": 0, "One year":1, "Two year": 2, "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3})


data = data[data.columns.drop(['customerID'], errors='ignore')]
x = data[data.columns.drop(['Churn'], errors='ignore')]
y = to_categorical(data['Churn'])


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)


model = Sequential()
model.add(Dense(18, input_shape=(19, ), activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

y_pred = model.predict(x_test)

y_outcome = np.argmax(y_pred, axis=1)

y_outcome