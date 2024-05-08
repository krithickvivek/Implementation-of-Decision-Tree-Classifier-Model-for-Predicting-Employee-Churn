# Ex-No:6 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

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
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```python

# Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
# Developed by: Krithick Vivekananda
# RegisterNumber:  212223240075
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
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
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### Dataset:
![alt text](318506137-8a055be6-9fb2-4c5f-a373-46722c6b07b7.png)
### Data.info()
![alt text](318506304-06002716-d302-4710-b6a4-3975cec7473b.png)
### Checking if null values are present:
![alt text](318506457-900fa2e9-72d0-47cd-a348-82f20420a7dd.png)
### Value_counts:
![alt text](318506563-3c241b96-51a7-4649-95a3-7d6070f24933.png)
### Dataset after encoding:
![alt text](318506705-62326632-3a96-43e4-8147-2b428e67470a.png)
### X_values:
![alt text](318506851-1ed10b6f-2283-4fcd-a747-2a70e55d7463.png)
### Accuracy:
![alt text](318506936-8a8668e6-5c17-4f10-9f29-bd3b2d80a926.png)
### dt.predict:
![alt text](318507231-7447bc4a-b766-4d43-8b56-4a4e914a5f35.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
