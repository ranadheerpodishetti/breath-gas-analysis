import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


path_A549=  "14.07. - Incubator/14.07. - Incubator/A549"
path_MDA=  "14.07. - Incubator/14.07. - Incubator/MDA"
path_FB = "09.07. - Incubator/09.07. - Incubator/FB"

A549 = os.listdir(path_A549)
MDA = os.listdir(path_MDA)
FB = os.listdir(path_FB)
print(A549)
#wine_data = pd.read_csv("D:/MASTERS/Melanie_6CP/14.07. - Incubator/14.07. - Incubator/"+ files[0])
#print(wine_data)


import  numpy
A549_1 = pd.read_csv("D:/MASTERS/Melanie_6CP/14.07. - Incubator/14.07. - Incubator/A549/"+ A549[1], sep = ';')
MDA_1 =  pd.read_csv("D:/MASTERS/Melanie_6CP/14.07. - Incubator/14.07. - Incubator/MDA/"+ MDA[1], sep = ';')
FB_1 = pd.read_csv("D:/MASTERS/Melanie_6CP/09.07. - Incubator/09.07. - Incubator/FB/"+ FB[1], sep = ';')

print(A549_1)
print(MDA_1)
print(FB_1)



A549_2 = A549_1.loc[(A549_1['Average_Mass']>=20) & (A549_1['Average_Mass']<200)]
A549_1 = A549_2.iloc[:,0:2]
print(A549_1)
MDA_2 = MDA_1.loc[(MDA_1['Average_Mass']>=20) & (MDA_1['Average_Mass']<200)]
MDA_1 = MDA_2.iloc[:,0:2]
print(MDA_1)
FB_2 = FB_1.loc[(FB_1['Average_Mass']>=20) & (FB_1['Average_Mass']<200)]
FB_1 = FB_2.iloc[:,0:2]
print(FB_1)

df = [A549_1,MDA_1,FB_1]

data = pd.concat(df)
print()
avg_mass =data.iloc[:, 0]
avg_current = data.iloc[:,1]
colors = ['A549', 'MDA', 'FB']
plt.scatter(A549_1['Average_Mass'], A549_1['Average'], c='r')
plt.scatter(MDA_1['Average_Mass'], MDA_1['Average'], c='g')
plt.scatter(FB_1['Average_Mass'], FB_1['Average'], c='b')
plt.legend(colors)
plt.show()


