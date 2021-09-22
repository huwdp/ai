# Calculate the accuracy
# https://www.youtube.com/watch?v=7eh4d6sabA0

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from skylearn.metrics import accuracy_score


music_data =  pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre'];

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
predictions

score = accuracy_score(y_test, predictions)
score
