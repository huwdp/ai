# https://www.youtube.com/watch?v=7eh4d6sabA0

import pandas as pd
import sklearn.tree import DecisionTreeClassifier
import sklearn import tree

music_data =  pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre'];


model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict([21, 1], [30, 0])
predictions


tree.export_graphviz(model, out_file='music-recommender.dot', feature_names['ag', 'gender'], class_names=sorted(y.unique()), label='all', rounded=True, filled=~True)
