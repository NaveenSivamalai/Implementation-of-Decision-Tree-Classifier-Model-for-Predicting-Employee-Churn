# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NAVEEN S
RegisterNumber:  212222110030
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
## Data head:
![276489296-a75903c7-f185-42e2-853a-276435f4e6f3](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/55a29eee-78e2-4631-905b-7e26272bbec5)

## Information:

![276489431-74603144-2086-4add-8001-5f0db6c08a4b](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/5cc512fd-d4ad-4617-bc66-b4aa1810f6d5)

## Null set:
![276489527-7b35f090-016e-4aa8-9ab2-d4a1b477f420](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/ba57151c-2022-4bc5-aa78-cb54368a3731)

## Value_counts():
![276489622-f94ef29d-fdee-4b48-a40d-4e17e51cdf2e](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/7fd23126-a0b5-4d5d-9f94-cfa9d8a45e48)

## Data head:
![276489691-56767095-20e1-4b9f-b44a-6885b53a7d33](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/108b77d7-0d72-4e3c-a53a-6bb667426d1e)

## x.head():
![276489792-bcdeb18e-03be-4f84-82a8-91cb909e29dc](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/a84f445d-f89c-4f52-ae59-ff670b6f199c)

## Data Prediction:
![276489903-f6950012-686f-4982-9b56-b66bbf238449](https://github.com/NaveenSivamalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123792574/5b14cc1f-f38a-49a9-88e1-6d894e1ec79a)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
