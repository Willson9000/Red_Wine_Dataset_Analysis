
#importing data visualization tools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

#reading Csv
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
#seperateing each value by semi collen and directing the use of columns
dataset = pd.read_csv("winequality-red.csv", delimiter=',', usecols = columns)

#dimensions and rows of dataset, aswell as small summary
firstrows = dataset.head()
print(firstrows)
print('')
lastrows = dataset.tail()
print(lastrows)
print('')
summary = dataset.describe()
print(summary)
print('')

#Distribution of different catagories
fig = plt.figure(figsize =(18,18))
ax=fig.gca()
dataset.hist(ax=ax,bins =30)
plt.show()

# Creates an 4x3 grid of subplots using GridSpec
fig = plt.figure(figsize=(12, 8)) # sets size of figure
gs = GridSpec(nrows=4, ncols=3, figure=fig)
# Loops through each column and creates a scatter plot of its values against the index
for i, column in enumerate(dataset.columns):
    row, col = i // 3, i % 3
    ax = fig.add_subplot(gs[row, col])
    ax.scatter(dataset.index, dataset[column], s=3, alpha=0.3)
    ax.set_title(column)
#creates padding for inbetween each figure/plot
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

#runs correclation function agains dataset
relations = dataset.corr()
print(relations)
#visualises the correclation matrix using a heat map
fig = plt.figure(figsize=(12, 10)) # sets size of figure
sns.heatmap(relations, annot=True, cmap="Greens")
plt.show()