# Deployment
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Parameters
C = 10
epsilon = 0.01
model_output_file = "model.bin"


# evaluation function
def eval(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return round(rmse, 3), round(r2, 3)

# 1. Data preparation
df = pd.read_csv('./data/car_insurance_premium_dataset.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 3. Selecting features and target variable
df_train, df_val = train_test_split(df, test_size=0.2, random_state=8)
X_train = df_train.drop(columns='insurance_premium_($)')
y_train = df_train['insurance_premium_($)'].values

X_val = df_val.drop(columns='insurance_premium_($)')
y_val = df_val['insurance_premium_($)'].values

scaler_x = StandardScaler()
scaler_x.fit(X_train)

X_train_scaled = scaler_x.transform(X_train)
X_val_scaled = scaler_x.transform(X_val)

scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))


# 4. Model Training
print("Training the model")
model = SVR(C=C, epsilon=epsilon)
model.fit(X_train_scaled, y_train_scaled.ravel())

# model prediction
y_pred_scaled = model.predict(X_val_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()


rmse, r2 = eval(y_val, y_pred)
print(f"RMSE: {rmse}; r2 score: {r2}")


# 6. Save the model
with open(model_output_file, "wb") as f_out:
    pickle.dump((scaler_x, scaler_y, model), f_out)

print(f"The model is saved to {model_output_file}")
