import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.formula.api import ols



annaova_df = pd.DataFrame(columns =['Fvalue','Pvalue'])
path_A549=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
path_MDA=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
path_list = [path_A549,path_MDA,path_FB]
A549 = os.listdir(path_A549)
MDA = os.listdir(path_MDA)
FB = os.listdir(path_FB)
label = ['A549','MDA','FB']
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
        j += 1
        df_list.append(file)

plot_list =[]
for peak in range(100,10000,500):
    total_df = pd.concat(df_list).reset_index(drop=True)
    mass_per_charge = total_df.iloc[:,0]
    ion_intensity= total_df.iloc[:,1]
    peaks, indices = find_peaks(ion_intensity, height=peak )
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
        peaks_files, indices_files = find_peaks(ion_intensity_files, height=peak )
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

    DF_data = pd.concat(final_data_list).reset_index(drop=True)
    DF_data = DF_data.drop(['label'],axis =1)
    print(DF_data)
    import scipy.stats as stats

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(*DF_data.to_numpy().T)

    df2 = pd.DataFrame({'Fvalue': [fvalue],
                        'Pvalue': [pvalue],
                         })

    annaova_df = pd.concat([annaova_df, df2], ignore_index=True, axis=0)

    print(annaova_df)
    annaova_df.to_csv('Annova.xlsx')