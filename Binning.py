import glob
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
sns.set()
from scipy.signal import find_peaks

from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from sklearn.decomposition import PCA #Principal Component Analysis

path_A549=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
path_MDA=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
path_list = [path_A549,path_MDA,path_FB]
A549 = os.listdir(path_A549)
MDA = os.listdir(path_MDA)
FB = os.listdir(path_FB)
label = [0,1,2]
list_1 = [A549,MDA,FB]
df_list = []
for i in range(0,3):
    path =  path_list[i]
    j = 0
    for file in list_1[i]:

        file = pd.read_csv(path+str(file),sep=';')
        file['class'] = label[i]
        file =file.drop(columns = ['Current_Mass','Current'])
        file = file.loc[(file['Average_Mass']> 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
        #print(file)
        x_data = file['Average_Mass']
        y_data = file['Average']

        from scipy.stats import binned_statistic

        x_bins, bin_edges, misc = binned_statistic(x_data, y_data, statistic="max", bins=100)

        bin_intervals = pd.IntervalIndex.from_arrays(bin_edges[:-1], bin_edges[1:])
        x_bins = pd.DataFrame(np.transpose(x_bins))
        x_bins =x_bins.transpose()
        x_bins['label'] = file.iat[0,2]
        df_list.append(x_bins)

data = pd.concat(df_list).reset_index(drop=True)

def set_to_median(x, bin_intervals):
    for interval in bin_intervals:
        if x in interval:
            return interval.mid




data1 = data.loc[:, data.columns != 'label']

data1 = pd.DataFrame(scaler.fit_transform(data1))

kmeans = KMeans(n_clusters=3)

kmeans.fit(data1)
#
#Find which cluster each data-point belongs to

clusters = kmeans.predict(data1)

data["Cluster"] = clusters

print(data)

comparison_column = np.where(data["label"] == data["Cluster"], 1, 0)

print(sum(comparison_column))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data1)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['label']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()