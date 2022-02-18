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
        file = file.loc[(file['Average_Mass']> 20) & (file['Average_Mass'] < 200)].reset_index(drop=True)
        j += 1
        df_list.append(file)

total_df = pd.concat(df_list).reset_index(drop=True)
mass_per_charge = total_df.iloc[:,0]
ion_intensity= total_df.iloc[:,1]
peaks, indices = find_peaks(ion_intensity, height=250)
mass_per_charge = mass_per_charge.take(peaks).to_frame()
mass_per_charge = mass_per_charge.sort_values(by=['Average_Mass'])
mass_per_charge = mass_per_charge.reset_index(drop=True)


mass_per_charge_list = mass_per_charge['Average_Mass'].values.tolist()
#print(mass_per_charge_list)
attribute = []
pop_count = 0
duplicate_1 = mass_per_charge_list.copy()
duplicate_2 = mass_per_charge_list.copy()
for item_1 in mass_per_charge_list:

    for item_2 in duplicate_1:

        if abs(item_1-item_2)<=0.2 and abs(item_1-item_2)!=0:
             if item_2 in duplicate_2:
                duplicate_2.remove(item_2)

    duplicate_1.pop(0)

final_mz = np.unique(duplicate_2).tolist()
#print(final_mz)

final_data_list = []

for files in  df_list:
    mass_per_charge_files = files.iloc[:, 0]
    ion_intensity_files = files.iloc[:, 1]
    peaks_files, indices_files = find_peaks(ion_intensity_files, height=250)
    mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
    #mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
    #mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

    mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

    peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

    file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
    #print(file)
    mz_1 = list(file['Average_Mass'])
    peak_heights = list(file['peak_heights'])
    mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

    mz_df_1['ion_intensity'] = 0



    for i in final_mz:
        for j in mz_1:
            if i <= j + 0.2:
                if abs(i - j) <= 0.2:
                    mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[file['Average_Mass'] == j, 'peak_heights'].item()

    mz_df_1 = list(mz_df_1.iloc[:,1])
    #print(mz_df_1)
    mz_df_1 = pd.DataFrame(mz_df_1).transpose()
    mz_df_1['label'] = files.iat[0,2]
    #print(mz_df_1)
    final_data_list.append(mz_df_1)

data = pd.concat(final_data_list).reset_index(drop=True)
print(data)

data1 = data.loc[:, data.columns != 'label']

data1 = pd.DataFrame(scaler.fit_transform(data1))

kmeans = KMeans(n_clusters=3)

kmeans.fit(data1)

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