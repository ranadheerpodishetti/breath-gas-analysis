
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1,
                               n_classes=3)
    return X, y


# get a list of models to evaluate
def get_models():
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        # create name for model
        key = '%.4f' % p
        # turn off penalty in some cases
        if p == 0.0:
            # no penalty in this case
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
        else:
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits = 4, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


import os
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

scaler = StandardScaler()

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier


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

    data = pd.concat(final_data_list).reset_index(drop=True)
    print(data)

    label_encoder = LabelEncoder()

    columns = list(data.columns.values)
    X_columns = columns[0:-1]
    Y_columns = [columns[-1]]
    print(Y_columns)

    X = data[X_columns].values
    Y = data[Y_columns].values

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X = StandardScaler().fit_transform(X)

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y ,train_size=0.80, test_size=0.20, random_state=101)
    print(f"for peak {peak}")
    # get the models to evaluate
    # training a DescisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0)

    scores = cross_val_score(clf, X, Y,scoring='accuracy', cv=4)
    print(scores)




