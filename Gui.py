import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

########################                                                      EDA                                         #################################
#importing data visualization tools

#reading Csv
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
#seperateing each value by semi collen and directing the use of columns
dataset = pd.read_csv("winequality-red.csv", delimiter=',', usecols = columns)

#dimensions and rows of dataset, aswell as small summary
firstrows = dataset.head()
lastrows = dataset.tail()
summary = dataset.describe()

#Distribution of different catagories
fig = plt.figure(figsize =(18,18))
ax=fig.gca()
dataset.hist(ax=ax,bins =30)

# Creates an 4x3 grid of subplots using GridSpec
fig2 = plt.figure(figsize=(12, 8)) # sets size of figure
gs = GridSpec(nrows=4, ncols=3, figure=fig)
# Loops through each column and creates a scatter plot of its values against the index
for i, column in enumerate(dataset.columns):
    row, col = i // 3, i % 3
    ax = fig.add_subplot(gs[row, col])
    ax.scatter(dataset.index, dataset[column], s=3, alpha=0.3)
    ax.set_title(column)
#creates padding for inbetween each figure/plot
plt.subplots_adjust(wspace=0.3, hspace=0.5)

#runs correclation function agains dataset
relations = dataset.corr()

#visualises the correclation matrix using a heat map
fig3 = plt.figure(figsize=(12, 10)) # sets size of figure
sns.heatmap(relations, annot=True, cmap="Greens")

########################                                                      PDA                                        ####################################

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

header = st.container()

st.title("Red Wine DataSet Analysis")
st.write(f"The exploratory Data Analysis for this report was completed in Visual Studio using several libraries and functions, it also utilized streamlit as a GUI for the EDA findings/answers.\n")

st.markdown("""
The exploratory Data Analysis for this report was completed in Visual Studio using several libraries and functions, it also utilized streamlit as a GUI for the EDA findings/answers.
The questions I attempted to answer follow as:

1.	What is the quality of data in the red wine data set?

2.	How do the different features in the dataset compare in terms of their distributions, and are there any noticeable trends or patterns?

3.	Are there any outliers or anomalies in the dataset that could affect our analysis or modelling?

4.	Which attributes of the wine effects its qualities favourable and unfavourably the most?

5.	Are there any strong correlations between the different chemical properties of the wine?

""")



st.header("First few rows of dataset")
st.write(firstrows)

st.header("Last few rows of dataset")
st.write(lastrows)

st.header("summary of dataset")
st.write(summary)

st.markdown("""
These resulting tables represent the quality of the data as well as the summary, 
from it I can decern that the dataset is consistent in the data type and quality, 
not having any random unreadable data, as well as having one value for every single row with none missing.
""")
st.header("Distribution of attributes")
st.write(fig)
st.markdown("""
These graphs show the distribution of all data categories of which you can see have no features or trends that stand out, 
though it can be seen that the data set has a high number of mediocre wine quality
""")
st.header("Correlation Matrix Heat Map")
st.pyplot(fig3)
st.markdown("""
Looking at the correlation matrix it can be concluded that alcohol (0.48), 
sulphates (0.25), and citric acid (0.23) in that order are the most favourable qualities in red wine and contribute most to its quality rating. 
The opposite would be the Volatile acidity (-0.39), total sulphur dioxide (-0.19) and density (-0.17), in that order. It could be assumed that higher qualities of red wine is rated mostly based on the former three qualities. 

The correlation matrix also shows the two attributes which are most correlated to each other, 
which are the pH levels and fixed acidity which have an absolute value of 0.682978, or expectedly free sulphur dioxide and total sulphur dioxide, 
though the only meaningful connection between the any variables that doesn’t include quality would be that the density of the red wine, which was effected greatly by the fixed acidity levels having a correlation of 0.67, 
though other than those few high correlations there wasn’t any interesting trend or connection visible.

""")
st.header("Model Accuracy Results")

#iterates through models list, name and model.
for name, model in models:
    kfold = StratifiedKFold(n_splits=9, random_state=1, shuffle=True) # splits data 9 times randomly, and shuffles.
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') # runs cross validation and provides scores
    results.append(cv_results)
    names.append(name)
    accuracy = cv_results.mean()
    # prints accuracy results
    st.write(f"{name}: {accuracy}")
st.write('LDA = Linear Discriminant Analysis')
st.write('Dtree = Decision Tree Classifie')
st.write('Rtree = Random Tree Classifier')

fig4, ax = plt.subplots()
ax.boxplot(results, labels=names) # Different algorthim accuracy tests.
ax.set_title('Algorithm Accuracy Comparison')

st.header("Model Accuracy Results Figure")
st.pyplot(fig4)
st.markdown("""
This shows that the Random Tree Classifier Model is the most promising model for the red wine data set from those chosen, 
though it also reveals that the Linear Discriminant Analysis has an outlier for a strange reason, this could be due to the way an LDA algorithm works functions.

Next the Random Tree Classifier Model is run against the entire training dataset that was created.
""")

# Runs training data on RandomForestClassifier method
model = RandomForestClassifier()
model.fit(X_train, Y_train)
# runs prediction model against x validation
predictions = model.predict(X_validation)

from sklearn.metrics import classification_report

# makes report on Y_validation to X validation
report = classification_report(Y_validation, predictions)
st.header("Rtree classifier Results agiasnt whole dataset")
st.text(report)
st.markdown("""
From this We can ascertain that the Random Tree Classifier model that we made was ~72% accurate at predicting the quality rating of red wine, depending on its makeup.
""")

st.header("Conclusion")
st.markdown("""
This Report demonstrates the Research and implementation done for the capstone project for ST1, 
it demonstrates the EDA of the red wine dataset using a multitude of methods to answer the EDA questions, 
it shows the PDA work done with the red wine dataset, allowing the user to predict the quality of red wine to an accuracy of 71%, 
it demonstrates  that machine learning algorithms can be used to accurately predict ratings and events to a degree ,
equal to human experts, with far less time and cost. 

The report demonstrates that with greater dataset size, 
better and more descriptive data, and better machine learning algorithms, 
accuracy results can be greatly increased to near perfect prediction accuracy.
""")

link = '[GitHub Repository](https://github.com/Willson9000/Red_Wine_Dataset_Analysis)'
st.markdown(link, unsafe_allow_html=True)
