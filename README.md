# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer.

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

8.Find the accuracy of the model.

9.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RUDESH KANNA R
RegisterNumber:  212223233002
*/
```
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('/content/spam.csv', encoding='latin-1')
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df[['label', 'text']]

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
```
## Output:
![image](https://github.com/user-attachments/assets/59642f0e-8efa-443d-b583-1357fc74778c)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
