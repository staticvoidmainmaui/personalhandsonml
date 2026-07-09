import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

#Load the data
oecd_bli =pd.read.csv("oecd_bli_2015.csv", thousands =',')
gdp_per_capita= pd.read_csv("gdp_per_capita.csv", thousands='',
                            delimiter='t', encoding='latin1', na_values="n/a")

#Prepare the data
country_stats= prepare_country_stats(oecd_bli, gpd_per_capita)
X= np.c_[country_stats["GDP per capita"]]
y=np.c[country_stats["life satisfaction"]]

#Visualize the data
country_stats.plot(kind='scatter', x= "GDP per capita", y= "Life satisfaction")
plt.show()

#Select linear model
model=sklearn.linear_model.LinearRegression()

#Train Model
model.fit(X,y)

#Make a prediction for Cyprus
X_new=[[22587]] # Cyprus GDP per capita4
print(model.predict(X_new)) #Outputs [5.553085] something outputted idk

#make claude - fix this up as an example and pull data so i can actually visualize what im looking at and working on

