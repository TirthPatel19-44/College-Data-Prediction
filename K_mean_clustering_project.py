import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('College_Data.csv', index_col=0)
# print(df.head())
# print(df.info())
# print(df.describe())

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data = df, hue = 'Private',
           palette = 'coolwarm', size = 6, aspect = 1, fit_reg = False)

sns.set_style('darkgrid')
g = sns.FacetGrid(df, hue = "Private", palette = 'coolwarm', size = 6, aspect = 2)
g = g.map(plt.hist,'Outstate',bins = 20, alpha = 0.7)

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
plt.show()
# ## K Means Cluster Creation
#
# Now it is time to create the Cluster labels!
#
# ** Import KMeans from SciKit Learn.**

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)
kmeans.fit(df.drop('Private', axis = 1))

kmeans.cluster_centers_

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

