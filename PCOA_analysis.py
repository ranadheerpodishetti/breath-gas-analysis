import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
from scipy.signal import find_peaks

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.preprocessing import LabelEncoder, StandardScaler

import sklearn.model_selection as model_selection




path_A549=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
path_MDA=  'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
path_A549_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549_stress/'
path_MDA_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA_stress/'
path_list = [path_A549,path_MDA,path_FB,path_A549_stress,path_MDA_stress]
A549 = os.listdir(path_A549)
MDA = os.listdir(path_MDA)
FB = os.listdir(path_FB)
A549_stress = os.listdir(path_A549_stress)
MDA_stress = os.listdir(path_MDA_stress)

label = ['A549','MDA','FB','A549_stress','MDA_stress']
list_1 = [A549,MDA,FB,A549_stress,MDA_stress]
df_list = []
for i in range(0,5):
    path =  path_list[i]
    j = 0
    for file in list_1[i]:

        file = pd.read_csv(path+str(file),sep=';')
        file['class'] = label[i]
        file =file.drop(columns = ['Current_Mass','Current'])
        file = file.loc[(file['Average_Mass']> 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
        j += 1
        df_list.append(file)


def pcoa_plot(DF_data):
    print(DF_data)
    # DF_corr = pd.DataFrame(np.corrcoef(DF_data.T),
    #                        index=DF_data.columns,
    #                        columns=DF_data.columns)
    # DF_corr.replace([np.inf, -np.inf], np.nan)
    # DF_corr = DF_corr.replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
    # # Eigendecomposition
    # Ve_eig_vals, Ar_eig_vecs = np.linalg.eig(DF_corr)
    #
    # # Sorting eigenpairs
    # eig_pairs = [(np.fabs(Ve_eig_vals[j]), Ar_eig_vecs[:, j]) for j in range(DF_data.shape[1])]
    # eig_pairs.sort()
    # eig_pairs.reverse()
    # Ar_components = np.array([x[1] for x in eig_pairs])
    #
    # # Projection matrix
    # Ar_Wproj = np.array([x[1] for x in eig_pairs]).T
    #
    # DF_transformed = pd.DataFrame(np.matmul(DF_data.values, Ar_Wproj),
    #                               columns=["PC_%d" % k for k in range(1, Ar_Wproj.shape[1] + 1)])

    from scipy.spatial import distance

    Ar_MxMdistance = distance.squareform(distance.pdist(DF_data.T , metric="braycurtis"))
    DF_dism = pd.DataFrame(Ar_MxMdistance, index=DF_data.columns, columns=DF_data.columns)
    print(DF_dism.to_string())
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(DF_dism)
    # plt.title(f"Matrix plot for peak: {peak}")
    # plt.colorbar()
    # #plt.show()
    DF_dism = DF_dism.fillna(0)
    import skbio

    # Compute the Principal Coordinates Analysis
    my_pcoa = skbio.stats.ordination.pcoa(DF_dism.values )

    # Show the new coordinates for our cities
    return my_pcoa




plot_list =[]
for peak in range(100,1000,500):
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
    print(DF_data)
    DF_data = DF_data.drop(['label'],axis =1)
    print(DF_data)
    data = DF_data.iloc[0:24, :]
    print(data)
    A549_pcoa = pcoa_plot(DF_data.iloc[0:24,:])
    MDA_pcoa = pcoa_plot(DF_data.iloc[24:53,:])
    FB_pcoa = pcoa_plot(DF_data.iloc[53:71,:])
    A549_stress_pcoa = pcoa_plot(DF_data.iloc[71:89,:])
    MDA_stress_pcoa = pcoa_plot(DF_data.iloc[89:107,:])

    #print(my_pcoa.samples[['PC1', 'PC2']])

    import matplotlib.pyplot as plt

    plt.scatter(A549_pcoa.samples['PC1'], A549_pcoa.samples['PC2'],marker = 'o')
    plt.scatter(MDA_pcoa.samples['PC1'], MDA_pcoa.samples['PC2'],marker = 'v')
    plt.scatter(FB_pcoa.samples['PC1'], FB_pcoa.samples['PC2'],marker = 'x')
    plt.scatter(A549_stress_pcoa.samples['PC1'], A549_stress_pcoa.samples['PC2'],marker = '+')
    plt.scatter(MDA_stress_pcoa.samples['PC1'], A549_stress_pcoa.samples['PC2'],marker = '*')
    plt.legend(label)
    plt.show()
    # for i in range(len(DF_dism.columns)):
    #     plt.text(my_pcoa.samples.loc[str(i), 'PC1'], my_pcoa.samples.loc[str(i), 'PC2'], DF_dism.columns[i])

    #plt.show()
