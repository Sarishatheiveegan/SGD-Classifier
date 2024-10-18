# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. 1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MARINO SARISHA T
RegisterNumber:  212223240084
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
x
y=df['target']
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
print("Prediction")
y_pred
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.3f}")
confusion=confusion_matrix(y_test,y_pred)
print("confusion matrix")
confusion
```

## Output:
![Screenshot 2024-10-18 110509](https://github.com/user-attachments/assets/31894a0a-7448-4fe7-926a-83903ffcedda)
![Screenshot 2024-10-18 110520](https://github.com/user-attachments/assets/8212dac3-865e-4383-b84c-876362c276bb)
![Screenshot 2024-10-18 110528](https://github.com/user-attachments/assets/f3d3f548-cc27-44e4-b0f3-786792dab12e)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
