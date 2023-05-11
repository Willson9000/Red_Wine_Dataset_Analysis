
#importing data visualization tools
import pandas as pd
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
dataset = pd.read_csv("winequality-red.csv", delimiter=',', usecols = columns)

from dataclasses import dataclass

@dataclass(eq=True, frozen=True, order=True) # class for wine that defines and sets attributes to the float
# frozen doesnt allow any changes to the objects will raise an error
# order stops other methods from being raised
class Wine:
    fixed_acidity : float
    volatile_acidity : float
    citric_acid : float
    residual_sugar : float
    chlorides : float
    free_sulfur_dioxide : float
    total_sulfur_dioxide : float
    density : float
    pH : float
    sulphates : float
    alcohol : float
    quality : float

#creates an empty list of redWines that will store instances of the redWine class, setting every attriibute to flaot.
Wines: list[Wine] = [] 
for index, item in dataset.iterrows():
    Wines.append(Wine(item['fixed acidity'],item['volatile acidity'],item['citric acid'],item['residual sugar'],item['chlorides'],item['free sulfur dioxide'],item['total sulfur dioxide'],item['density'],item['pH'],item['sulphates'],item['alcohol'],item['quality']))

#creating empty lists for quality and attribute data to go in.
x = []
y = []

from sklearn.model_selection import train_test_split

#seperates the quality and attributes data into y and x list
for wine in Wines:
    x.append([wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid, wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide, wine.total_sulfur_dioxide, wine.density, wine.pH, wine.sulphates, wine.alcohol])
    y.append(wine.quality)

# seperates 20% of the both tables data is randomly seperated for validation  and the rest is left for training
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#list of models and names to itterate through
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('Dtree', DecisionTreeClassifier()))
models.append(('Rtree', RandomForestClassifier()))
#results and names lists
results = []
names = []

#cross validation test
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#iterates through models list, name and model.
for name, model in models:
	kfold = StratifiedKFold(n_splits=9, random_state=1, shuffle=True) # splits data 9 times randomly, and shuffles.
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') # runs cross validation and provides scores
	results.append(cv_results)
	names.append(name)
	# prints accuracy results
	print(f"{name}: {cv_results.mean()}")


import matplotlib.pyplot as plt
plt.boxplot(results, labels=names) # Different algorthim accuracy tests.
plt.title('Algorithm Accuracy Comparison')
plt.show()


# Runs training data on RandomForestClassifier method
model = RandomForestClassifier()
model.fit(X_train, Y_train)
# runs prediction model against x validation
predictions = model.predict(X_validation)

from sklearn.metrics import classification_report

# makes report on Y_validation to X validation
report = classification_report(Y_validation, predictions)
print(report)

import numpy as np

#user input function
def user():
    try:
        # ask user input for a string
        print("Valid Data e.g: 10.4;0.41;0.55;3.2;0.076;22;54;0.9996;3.15;0.89;9.9 (only 11)")
        n = input("Enter valid data in the format: ")
        # removes semi colon from string, inserts into 2d array and stores value in variable
        user_val = np.array([[float(k) for k in i.split(';')] for i in n.splitlines()])
        # runs prediction model against user input, gives quality rating
        print(model.predict(user_val))
        print("out of [9]")
    except ValueError:
        print("Error Please enter a string")
        return
    
user()