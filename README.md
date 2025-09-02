# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KATHI HASINI
RegisterNumber:  212224240074
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

filename = "markml.csv"

if not os.path.exists(filename):
    print(f"{filename} not found. Creating sample dataset...")
    sample_data = {
        "Hours": [1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0],
        "Scores": [20, 30, 35, 40, 45, 50, 60, 65, 70, 85, 95]
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(filename, index=False)
    print(f"Sample dataset '{filename}' created successfully!")

df = pd.read_csv(filename)

print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())

X = df.iloc[:, :-1].values  # Hours (input)
Y = df.iloc[:, -1].values   # Scores (output)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("\nPredicted Scores:", Y_pred)
print("Actual Scores:", Y_test)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")  # Line from training set
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)
```

## Output:
<img width="926" height="855" alt="image" src="https://github.com/user-attachments/assets/d59eaf7f-fb8d-4534-ae12-8149f2e4a543" />
<img width="982" height="614" alt="Screenshot 2025-09-02 150919" src="https://github.com/user-attachments/assets/49bf4a57-bfc2-433a-b4af-09134aeb13db" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
