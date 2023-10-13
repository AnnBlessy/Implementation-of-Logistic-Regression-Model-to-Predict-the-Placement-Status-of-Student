# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ANN BLESSY PHILIPS
RegisterNumber:  212222040008
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print("Placement data")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print("print data")
data1

x = data1.iloc[:,:-1]
print("Data-status")
x

y = data1["status"]
print("data-status")
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![EXP4-1](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/c637c601-a65c-490a-b150-e59fd7ead20f)

![EXP4-2](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/0ca792ab-9331-49da-971f-942451d53a8a)

![EXP4-3](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/051f3e58-99cf-458c-9644-0c5d77a48060)

![EXP4-4](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/ba9a0a0d-73d7-44cb-8057-c78f80987cc9)

![EXP4-5](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/f58c4a69-7509-475b-97f1-e0561dbf99ec)

![EXP4-6](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/7134dfb3-7bc9-42c3-8b07-ad63f1bcd6a8)

![EXP4-7](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/31cdb53c-11bf-4150-8738-a72303ca0736)

![EXP4-8](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/e91e0f4c-a1c3-4b0e-8c32-d29cccdff8bf)

![EXP4-9](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/92d51ca5-dde9-4935-8493-c883a155e802)

![EXP4-10](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/9ce3f0df-2f98-4e70-ab51-09367c9a5aa4)

![EXP4-11](https://github.com/AnnBlessy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477835/dfc725d5-f0d5-4cd5-9946-98bb5af4f231)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
