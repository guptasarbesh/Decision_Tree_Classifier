import pandas as pd
import numpy as np

#from sklearn import datasets
file1=pd.read_csv('D:\SEMESTER VI\MACHINE LEARNNING FOUNDATION CODES\dermatology.csv',sep=',')
print(file1.describe())
print(file1.head())
print(file1.tail())
print(file1.columns)
print(file1.index)
#print(file1.erythema)
#print(file1.scaling)
print(type(file1))  
print(file1.shape)
import matplotlib.pyplot as plt
file1.plot()
plt.show()


print(file1.isnull().sum())


col_name=['erythema', 'scaling', 'definite_borders', 'itching',
       'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
       'oral_mucosal_involvement', 'knee_and_elbow_involvement',
       'scalp_involvement', 'family_history', 'melanin_incontinence',
       'eosinophils_in_the_infiltrate', 'pnl_infiltrate',
       'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
       'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
       'elongation_of_the_rete_ridges',
       'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule',
       'munro_microabcess', 'focal_hypergranulosis',
       'disappearance_of_the_granular_layer',
       'vacuolisation_and_damage_of_basal_layer', 'spongiosis',
       'saw-tooth_appearance_of_retes', 'follicular_horn_plug',
       'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
       'band-like_infiltrate', 'age', 'class']
file1.columns=col_name
print(file1.head())

#file1=file1.astype(float)
#print("Object are changed into numeric")
#print(file1.dtypes)

file1.fillna(file1.median(),inplace=True)
print(file1.head(20))

print(file1.isnull().sum())
y=file1['class']
x=file1.drop('class',axis=1)
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

print(x_train.shape)
print(x_test.shape)

from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test,check_input=True)
print('misclassified samples:%d'%(y_test!=y_pred).sum())

##
from sklearn.metrics import accuracy_score
print('Accuracy %.2f'%accuracy_score(y_test,y_pred))

##
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_dist={'max_depth':np.arange(1,100,1),'criterion':['gini','entropy']}
dc=DecisionTreeClassifier(random_state=2)
cv=GridSearchCV(dc,param_dist,cv=5)
cv.fit(x_train,y_train)
print(cv.best_params_)
print(cv.best_score_)
