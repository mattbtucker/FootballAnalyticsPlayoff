import pandas as pd
import numpy as np
from sklearn import preprocessing
from xgboost import XGBClassifier
from matplotlib import pyplot

#Import Features and Outcomes Data
rawtable = pd.read_csv('/Users/matttucker/Desktop/NFLCSV.csv')
outcomes = pd.read_csv('/Users/matttucker/Desktop/NFLCSVOutcomesVector.csv')

#Changes to array
outcomes = np.ravel(outcomes)
x = rawtable.values 

#Scale data for processing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

#Import scaled data into data frame
df = pd.DataFrame(x_scaled)

#Define model and fit
model = XGBClassifier()
model.fit(df, outcomes)

#plot feature importance for classifying teams
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.title('NFL Offensive Statistics and Relative Importance for Predicting Playoff Teams')
pyplot.show()

##TODO
#Inestigate model robustness by training and testing (cross-validation due to limited data)
#Deploy on previous years to test ability to project playoff teams
