# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5.Determine training and test data set.
6.Apply decision tree Classifier on to the dataframe
7.Get the values of accuracy and data prediction.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SURIYA PRAKASH.S
RegisterNumber:  212223100055

```

```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head(5)

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data['left']
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print("SURIYA PRAKASH.S[212223100055]")
```

## Output:
### Initial data set:
![image](https://github.com/user-attachments/assets/0c0a8442-741d-4a7f-b3b5-5cbdea98da95)


### Optimization of null values:
![image](https://github.com/user-attachments/assets/293715dc-0c0c-4164-90d1-e4268277c950)


### Assignment of x and y values:

![image](https://github.com/user-attachments/assets/c99ae0c9-3d47-45fd-99e5-75c6ccf6750e)

![image](https://github.com/user-attachments/assets/595d6cb6-3898-4696-a9d3-6c1fd890ccbc)


### Converting string literals to numerical values using label encoder:

![image](https://github.com/user-attachments/assets/e76de8ea-e019-4e85-bc31-2d703eb6bf8e)

### Accuracy:
![image](https://github.com/user-attachments/assets/9b4c1df0-9f97-44b7-9d5f-79be73606ad6)


### Prediction:
![image](https://github.com/user-attachments/assets/36338e5e-8efe-410e-8fb2-97dc0ebb07c4)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
