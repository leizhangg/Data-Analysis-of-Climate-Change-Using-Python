# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:58:52 2020

@author: AK
"""
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("annual_consumption_state.csv")
#Step 1
print ("Reading file...")
print ("Electricity Consumption by State 1990-2019")
#displaying number of rows and columns in the dataframe
print("Number of rows in the dataframe: " + str(len(df)))
print("Number of columns in the dataframe: " + str(len(df.columns)))
print()
print("Data cleaning being performed...")
#Some values in 'CONSUMPTION_for_ELECTRICITY' have a '.', needs to be replaced with 0.
print()
df["CONSUMPTION_for_ELECTRICITY"].replace({".": "0"}, inplace=True)
#Decimal valuea are rounded off to 4 digits
df = df.round(4)
#The 'US-TOTAL' row is excluded 
df = df[df.STATE != 'US-TOTAL']
#The 'US-Total' row is excluded 
df = df[df.STATE != 'US-Total']
#The column 'CONSUMPTION_for_ELECTRICITY' is a string, but it contains numbers.
#Hence it is converted to depict numeric values
df[['CONSUMPTION_for_ELECTRICITY']] = df[['CONSUMPTION_for_ELECTRICITY']].apply(pd.to_numeric) 

df.describe()
#The amount of electricity consumed by every state is calculated and stored in dataframe named 'stateVsConsumption'
stateVsConsumption = df.groupby(["STATE"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
#The column 'CONSUMPTION_for_ELECTRICITY' is converted to an integer type.
stateVsConsumption['CONSUMPTION_for_ELECTRICITY'] = stateVsConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#The values in the dataframe 'stateVsConsumption' are sorted based on column 'CONSUMPTION_for_ELECTRICITY'
stateVsConsumption = stateVsConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')

#for plotting purpose, all the unique state names are obtained and stored in a list named 'stateName'
stateName = list(stateVsConsumption.STATE.unique()) 
#The electricity consumed is the column 'CONSUMPTION_for_ELECTRICITY' which is converted to a list and stored in the list named 'consumption'
consumption = stateVsConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
#Plot the graph
fig, ax = plt.subplots(figsize=(12, 8))
#the color of the line graph can be changed here- it is yellow because my third parameter is 'y'
ax.plot(stateName, consumption, 'y')
#ifline 48 is commented, the background color will be white
ax.set_facecolor('xkcd:white')
ax.set_title('Electricity Consumption by State')
plt.xlabel('State') #the naming for x axis
plt.ylabel('Electricity Consumed')  #the namong for y-axis
plt.xticks(rotation=90) #so that the names of states are readable and don't overlap, change the angle
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()
stateVsConsumption = stateVsConsumption.sort_values('CONSUMPTION_for_ELECTRICITY', ascending = False)

print("The bottom three states with the lowest combined energy source consumption for electrcity 1990-2019:")
#the top three values are obtained using the iloc function and indexing
bottom3 = stateVsConsumption.tail(3)
#it is displayed on the screen without printing the index of these rows
print(bottom3.to_string(index = False))
print()
print("The top three states with the highest combined energy source consumption for electrcity 1990-2019:")
#the bottom three values are obtained using the 'tail' function
top3 = stateVsConsumption.iloc[:3]
#it is displayed on the screen without printing the index of these rows
print(top3.to_string(index = False))

#to get the electricity consumption based on every year
yearVsConsumption = df.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
#The electricity consumed is the column 'CONSUMPTION_for_ELECTRICITY' which is converted to type int
yearVsConsumption['CONSUMPTION_for_ELECTRICITY'] = yearVsConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#for plotting purpose, all the unique year values are obtained and stored in a list named 'year'
year = list(yearVsConsumption.YEAR.unique()) 
#The electricity consumed is the column 'CONSUMPTION_for_ELECTRICITY' which is converted to a list and stored in the list named 'consumption'
consumption = yearVsConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
#the figure is plotted, the size of the figure is also specified
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumption, 'g')
ax.set_facecolor('xkcd:white')
ax.set_title('Electricity Consumption By Year 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity consumed')  #the naming for y-axis
#plt.xticks(rotation=45) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

print()
# print("The year in which lowest amount of electricity was consumed is :")
# print(year[0])
# print("The year in which highest amount of electricity was consumed is :")
# print(year[-1])

#to find the sum of electricity consumed based on the source of energy
#it is stored in a dataframe named'totalEnergy'
totalEnergy = df.groupby(["ENERGY_SOURCE"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
#The column 'CONSUMPTION_for_ELECTRICITY' is converted to an integer type.
totalEnergy['CONSUMPTION_for_ELECTRICITY'] = totalEnergy['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#The values in the dataframe 'totalEnergy' are sorted based on column 'CONSUMPTION_for_ELECTRICITY'
totalEnergy = totalEnergy.sort_values('CONSUMPTION_for_ELECTRICITY')
#the 'Other Gases (Billion BTU)' and 'Other Gases (Billion Btu)' contain same type of data- the two names mean the same
#Hence they are merged to indicate 'Other Gases (Billion BTU)' only
totalEnergy = totalEnergy.groupby(totalEnergy['ENERGY_SOURCE'].str[:5])['CONSUMPTION_for_ELECTRICITY'].sum().reset_index()
#Since first 5 letters would have been extracted (above line .str[:5]), the columns are renamed
totalEnergy['ENERGY_SOURCE'] = totalEnergy['ENERGY_SOURCE'].replace(
        {'Coal': 'Coal (Short Tons)', 
         'Geoth': 'Geothermal (Billion Btu)', 
         'Natur': 'Natural Gas (Mcf)', 
         'Other': 'Other Gases (Billion BTU)',
         'Petro' : 'Petroleum (Barrels)'})
#for plotting purpose, all the unique energy source types are obtained and stored in a list named 'energySource'
energySource = list(totalEnergy.ENERGY_SOURCE.unique()) 
#The electricity consumed is the column 'CONSUMPTION_for_ELECTRICITY' which is converted to a list and stored in the list named 'consumption'
consumption = totalEnergy['CONSUMPTION_for_ELECTRICITY'].tolist() 
#The size of the graph is specified and its colors are specified
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(energySource, consumption)
#ax.set_facecolor('xkcd:salmon')
ax.set_title('Electrcity Consumption by Energy Source')
plt.xlabel('Energy Source Type') #the naming for x axis
plt.ylabel('Electricity Consumption')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

totalEnergy = totalEnergy.sort_values('CONSUMPTION_for_ELECTRICITY', ascending = False)
print("The energy resource with maximum consumption of electricity is: ")
print(str(totalEnergy.iloc[0][0]) + " with " + str(totalEnergy.iloc[0][1]) +" units of consumption")
print()
print("The energy resource with minimum consumption of electricity is: ")
print(str(totalEnergy.iloc[len(totalEnergy)-1][0]) + " with " + str(totalEnergy.iloc[len(totalEnergy)-1][1]) +" units of consumption")
print()
##to display y-axis with scientific notation
#plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.xlabel('Energy resource type') #the naming for x axis
#plt.xticks(rotation=90)#so that the names of states are readable and don't overlap
#plt.title("Amount of electricity consumed by every energy resource")
#plt.ylabel('Electricity consumed')  #the namong for y-axis
#plt.bar(energySource, consumption) #plot the graph
#plt.show()

#the 'Other Gases (Billion BTU)' and 'Other Gases (Billion Btu)' contain same type of data- the two names mean the same
#Hence they are merged to indicate 'Other Gases (Billion BTU)' itslef in the original dataframe
df = df.replace(to_replace ="Other Gases (Billion Btu)", value ="Other Gases (Billion BTU)") 

#The amount of electricity consumed by specific energy source is calculated.
#All the 5 different types of energy resources are separately filtered out, stored in sepaarte dataframes
NaturalGasConsumption = df.loc[df['ENERGY_SOURCE'] == 'Natural Gas (Mcf)']
#The electricty consumed is found and stored in a dataframe
NaturalGasConsumption = NaturalGasConsumption.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
#the column 'CONSUMPTION_for_ELECTRICITY' data type is converted to 'int'
NaturalGasConsumption['CONSUMPTION_for_ELECTRICITY'] = NaturalGasConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#NaturalGasConsumption = NaturalGasConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')
#for plotting purpose, all the unique year values are obtained and stored in a list named 'year'
year = list(NaturalGasConsumption.YEAR.unique()) 
#The electricity consumed is the column 'CONSUMPTION_for_ELECTRICITY' which is converted to a list and stored in the list named 'consumption'
consumptionnaturalgas = NaturalGasConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumptionnaturalgas)
#ax.set_facecolor('xkcd:pink')
ax.set_title('Amount of Electricity Consumed by Natural Gas 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumption')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

CoalConsumption = df.loc[df['ENERGY_SOURCE'] == 'Coal (Short Tons)']
CoalConsumption = CoalConsumption.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
CoalConsumption['CONSUMPTION_for_ELECTRICITY'] = CoalConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#NaturalGasConsumption = NaturalGasConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')
year = list(CoalConsumption.YEAR.unique()) 
consumptioncoal = CoalConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumptioncoal)
#ax.set_facecolor('xkcd:light green')
ax.set_title('Amount of Electricity Consumed by Coal 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumed')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#energy resource here is petroleum
PetroleumConsumption = df.loc[df['ENERGY_SOURCE'] == 'Petroleum (Barrels)']
PetroleumConsumption = PetroleumConsumption.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
PetroleumConsumption['CONSUMPTION_for_ELECTRICITY'] = PetroleumConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#NaturalGasConsumption = NaturalGasConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')
year = list(PetroleumConsumption.YEAR.unique()) 
consumptionpetroleum = PetroleumConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumptionpetroleum)
#ax.set_facecolor('xkcd:light blue')
ax.set_title('Amount of Electricity Consumed by Petroleum 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumed')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#energy resource here is geothermal
GeothermalConsumption = df.loc[df['ENERGY_SOURCE'] == 'Geothermal (Billion Btu)']
GeothermalConsumption = GeothermalConsumption.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
GeothermalConsumption['CONSUMPTION_for_ELECTRICITY'] = GeothermalConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#NaturalGasConsumption = NaturalGasConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')
year = list(GeothermalConsumption.YEAR.unique()) 
consumptiongeothermal = GeothermalConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumptiongeothermal)
#ax.set_facecolor('xkcd:light yellow')
ax.set_title('Amount of Electricity Consumed by Geothermal 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumed')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#energy resource here is other gases
OtherGasesConsumption = df.loc[df['ENERGY_SOURCE'] == 'Other Gases (Billion BTU)']
OtherGasesConsumption = OtherGasesConsumption.groupby(["YEAR"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
OtherGasesConsumption['CONSUMPTION_for_ELECTRICITY'] = OtherGasesConsumption['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#NaturalGasConsumption = NaturalGasConsumption.sort_values('CONSUMPTION_for_ELECTRICITY')
year = list(OtherGasesConsumption.YEAR.unique()) 
consumptionothergases = OtherGasesConsumption['CONSUMPTION_for_ELECTRICITY'].tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(year, consumptionothergases)
#ax.set_facecolor('xkcd:light red')
ax.set_title('Amount of Electricity Consumed by Other Gases 1990-2019')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumed')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#function to generate values for geothermal resource, since it has data only from 2011
#it appends 0 to previous years
def Addzero(value):
    return [0] * (30-len(value)) + value
consumptiongeothermal = Addzero(consumptiongeothermal)

fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(year, consumption)
#ax.set_facecolor('xkcd:light pink')
ax.plot(year, consumptionnaturalgas, label = 'Natural Gas energy consumption')
ax.plot(year, consumptioncoal, label = 'Coal energy consumption')
ax.plot(year, consumptionpetroleum, label = 'Petroleum energy consumption')
ax.plot(year, consumptionothergases, label = 'Other Gases energy consumption')
ax.plot(year, consumptiongeothermal, label = 'Geothermal energy consumption')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large',
                   bbox_to_anchor=(0.5, -0.1))
legend.get_frame().set_facecolor('w')
ax.set_title('Amount of electricity consumed by various sources the years')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity consumed')  #the namong for y-axis
plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
plt.yticks([0,500000,100000,150000, 200000, 300000,400000,
            70000000,500000000,1000000000,1500000000,
            2000000000,7000000000,8000000000,
            10000000000,20000000000,30000000000])
ax.ticklabel_format(style = 'plain', axis = 'y')
ax.set_yscale('log')

plt.show()
###################
trainingSet_data = year
myVals = []
for i in range(2020, 2031,1):
    myVals.append(i)
validateSet_data = myVals
trainingSet_target_coal = consumptioncoal
trainingSet_target_naturalgas = consumptionnaturalgas
trainingSet_target_geothermal = consumptiongeothermal
trainingSet_target_petroleum = consumptionpetroleum
trainingSet_target_othergases = consumptionothergases

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
coalList=[]
geothermalList = []
naturalGasList = []
petroleumList = []   
otherGasList = []
def my_fun(trainingSet_data,trainingSet_target,resourcetype):
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    #reshaping is done so that there are no discrepancies while predicting values
    X = np.array(trainingSet_data).reshape(-1, 1)
    y = np.array(trainingSet_target)          
    #the values are fit to the model using the 'fit' function
    regr.fit(X, y) 
    # Train the model using the training sets
    X_new = np.array(validateSet_data).reshape(-1, 1)
    #the prediction is done using the 'predict' function
    y_pred = regr.predict(X_new)
    #it is converted to int type from array for displaying purposes
    y_pred_int = y_pred.astype('int64')
    print('Coefficients: \n', regr.coef_)
    for i in range(0,11):
            print("The predicted values of electricity consumption for " +str(resourcetype) + " in")
            print(str(validateSet_data[i]) + " is " + str(y_pred_int[i]) + " units")
            print()
    
    predictedVals_y = y_pred_int.tolist() 
    if resourcetype == 'Coal':
        coalList.append(predictedVals_y)
    if resourcetype == 'Geothermal':
        geothermalList.append(predictedVals_y)
    if resourcetype == 'Natural Gas':
        naturalGasList.append(predictedVals_y)
    if resourcetype == 'Petroleum':
        petroleumList.append(predictedVals_y)
    if resourcetype == 'Other Gas': 
        otherGasList.append(predictedVals_y)

my_fun(trainingSet_data, trainingSet_target_coal,'Coal')
my_fun(trainingSet_data, trainingSet_target_naturalgas,'Natural Gas')
my_fun(trainingSet_data, trainingSet_target_geothermal, 'Other Gas')
my_fun(trainingSet_data, trainingSet_target_petroleum, 'Petroleum')
my_fun(trainingSet_data, trainingSet_target_othergases, 'Geothermal')
   
import itertools

coalList = list(itertools.chain(*coalList))
petroleumList = list(itertools.chain(*petroleumList))
naturalGasList = list(itertools.chain(*naturalGasList))
otherGasList = list(itertools.chain(*otherGasList))
geothermalList = list(itertools.chain(*geothermalList))

petroleumList =  [abs(ele) for ele in petroleumList] 


fig, ax = plt.subplots(figsize=(12, 8))
#ax.plot(xVals, actualVals_y, label = 'Actual electricity consumed')
ax.scatter(myVals, coalList, label = 'Predicted electricity consumed by coal')
ax.scatter(myVals, petroleumList, label = 'Predicted electricity consumed by petroleum')
ax.scatter(myVals, naturalGasList, label = 'Predicted electricity consumed by natural gas')
#ax.scatter(myVals, otherGasList, label = 'Predicted electricity consumed by other gas')
#ax.scatter(myVals, geothermalList, label = 'Predicted electricity consumed by geothermal')
legend = ax.legend(loc='upper center', shadow=True, fontsize='large',
                   bbox_to_anchor=(0.5, -0.1))    
legend.get_frame().set_facecolor('w')
#ax.set_facecolor('xkcd:light pink')
ax.set_title('Predicted consumption of electricity')
plt.yticks([0,50000,100000,160000, 180000, 260000, 
            300000,400000,500000, 6000000,8000000,
            1000000,50000000, 70000000,
            90000000,100000000,1500000000,
            1700000000,20000000000,25000000000,
            30000000000])
ax.set_yscale('log')

plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity consumption')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
#ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()


###################
#based on the state, the electricity consumption is calculated and stored in dataframe 'HighestLowest
HighestLowest = df.groupby(["STATE","ENERGY_SOURCE"]).CONSUMPTION_for_ELECTRICITY.sum().reset_index()
#the datatype of 'CONSUMPTION_for_ELECTRICITY' is converted to int
HighestLowest['CONSUMPTION_for_ELECTRICITY'] = HighestLowest['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#the below code is used to display the electricity consumed by every resource in every state
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

my_df = HighestLowest.set_index(['STATE','ENERGY_SOURCE'])['CONSUMPTION_for_ELECTRICITY'].unstack()
my_df['Coal (Short Tons)'] = my_df['Coal (Short Tons)'].fillna(0) 
my_df['Geothermal (Billion Btu)'] = my_df['Geothermal (Billion Btu)'].fillna(0)
my_df['Natural Gas (Mcf)'] = my_df['Natural Gas (Mcf)'].fillna(0) 
my_df['Other Gases (Billion BTU)'] = my_df['Other Gases (Billion BTU)'].fillna(0) 
my_df['Petroleum (Barrels)'] = my_df['Petroleum (Barrels)'].fillna(0) 
 
my_df['Coal (Short Tons)'] = my_df['Coal (Short Tons)'].astype('int64')
my_df['Geothermal (Billion Btu)'] = my_df['Geothermal (Billion Btu)'].astype('int64')
my_df['Natural Gas (Mcf)'] = my_df['Natural Gas (Mcf)'].astype('int64')
my_df['Other Gases (Billion BTU)'] = my_df['Other Gases (Billion BTU)'].astype('int64')
my_df['Petroleum (Barrels)'] = my_df['Petroleum (Barrels)'].astype('int64')

maxValues = my_df.idxmax(axis=1)
vals = my_df.max(axis=1)
maxValsDf = pd.concat([maxValues, vals], axis=1).reset_index()
maxValsDf.columns = ['STATE','Resource_Type', 'Electricity_consumed']
maxValsDf["STATE"] = maxValsDf["STATE"] + '-' + maxValsDf["Resource_Type"] 
del maxValsDf['Resource_Type']

minValues = my_df.idxmin(axis=1)
vals_2 = my_df.min(axis=1)
minValsDf = pd.concat([minValues, vals_2], axis=1).reset_index()
minValsDf.columns = ['STATE','Resource_Type', 'Electricity_consumed']
minValsDf["STATE"] = minValsDf["STATE"] + '-' + minValsDf["Resource_Type"] 
del minValsDf['Resource_Type']

state = maxValsDf.STATE.values.tolist()
consumption = maxValsDf.Electricity_consumed.values.tolist()
fig, ax = plt.subplots(figsize=(20,15))
ax.plot(state, consumption, 'b')
#ax.set_facecolor('xkcd:light yellow')
ax.set_title('State and Energy Resource with the Highest Electricity Consumption')
plt.xlabel('State and energy resource') #the naming for x axis
plt.ylabel('Highest Amount of Electricity Consumed by Every Resource in Each State')  #the namong for y-axis

plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap

#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

fig, my_ax = plt.subplots(figsize=(20,15))
maxValsDf.plot(x ='STATE', y='Electricity_consumed', kind = 'bar',ax = my_ax)	
my_ax.set_title('State and Energy Resource with the Highest Electricity Consumption')
my_ax.set_ylabel('State and Energy Resource')
my_ax.set_xlabel('Highest Amount of Electricity Consumed by Every Resource in Each State')
plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap
#plt.xticks(rotation=90)
my_ax.ticklabel_format(style = 'plain', axis = 'y')

####min vals####
state_1 = minValsDf.STATE.values.tolist()
consumption_1 = minValsDf.Electricity_consumed.values.tolist()
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(state_1, consumption_1, 'b')
#ax.set_facecolor('xkcd:light yellow')
ax.set_title('State and Energy Resource with the Lowest Electricity Consumption')
plt.xlabel('State and Energy Resource') #the naming for x axis
plt.ylabel('Lowest Amount of Electricity Consumed by Every Resource in Each State')  #the namong for y-axis
plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()
#same as above but a bar chart
fig, my_ax = plt.subplots(figsize=(20,15))
minValsDf.plot(x ='STATE', y='Electricity_consumed', kind = 'bar',ax = my_ax)	
my_ax.set_title('State and Energy Resource with Lowest Electricity Consumption')
my_ax.set_ylabel('State and Energy Resource')
my_ax.set_xlabel('Lowest Amount of Electricity Consumed by Every Resource in Each State')
plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap
#plt.xticks(rotation=90)
my_ax.ticklabel_format(style = 'plain', axis = 'y')

#PREDICTION PART
#using a linear regression model
#a copy of the dataframe yearVsConsumption (that holds electricity consumed every year) is made
pred_df = yearVsConsumption[['YEAR', 'CONSUMPTION_for_ELECTRICITY']].copy()
#the datatype of 'CONSUMPTION_for_ELECTRICITY' is converted to int
pred_df['CONSUMPTION_for_ELECTRICITY'] = pred_df['CONSUMPTION_for_ELECTRICITY'].astype('int64')
#The data is divided into two parts- one for training, other for testing. 
#training data should be more for the predictions to be more accurate.
#we have taken 28 values out of 30 values as training data
#the 'trainingSet_data' is a list with 'YEAR' values which are mapped to values of the 'CONSUMPTION_for_ELECTRICITY' column
trainingSet_data = pred_df['YEAR'][:30]
#the remaining 2 values belong to test dataset

#The target values are those that are mapped to 'YEAR' column 
yearVals = []
for i in range(2020, 2031,1):
    yearVals.append(i)
validateSet_data = yearVals
#The target values are those that are mapped to 'YEAR' column 
trainingSet_target = pred_df['CONSUMPTION_for_ELECTRICITY'][:30]

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()
#reshaping is done so that there are no discrepancies while predicting values
X = np.array(trainingSet_data).reshape(-1, 1)
y = np.array(trainingSet_target)          
#the values are fit to the model using the 'fit' function
regr.fit(X, y) 
# Train the model using the training sets
X_new = np.array(validateSet_data).reshape(-1, 1)
#the prediction is done using the 'predict' function
y_pred = regr.predict(X_new)
#it is converted to int type from array for displaying purposes
y_pred_int = y_pred.astype('int64')
#to predict electricity consumed for the upcoming years, use the below line of code- line 315
# The coefficients are displayed
print()
print('Coefficients: \n', regr.coef_)
# The mean squared error is displayed
print()
for i in range(0,11):
        print("The predicted values of electricity consumption for ")
        print(str(validateSet_data[i]) + " is " + str(y_pred_int[i]) + " units")
        print()

xVals = validateSet_data
predictedVals_y = y_pred_int.tolist() 
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xVals, predictedVals_y, label = 'Prediction About Electricity Consumption')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
ax.scatter(xVals, predictedVals_y, label = 'Prediction About Electricity Consumption')

legend.get_frame().set_facecolor('w')
#Xax.set_facecolor('xkcd:light pink')
ax.set_title('Predicted Electricity Consumption')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Electricity Consumption')  #the namong for y-axis
#plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#Code #2
import pandas as pd 
import matplotlib.pyplot as plt
#data frame is labeled as df here
df = pd.read_csv("Energy Sources Emissions_Combined.csv")
print ("=================================")
print ("Running analysis on the second data set....")
print ("U.S. Energy-Related Carbon Dioxide Emissions by Energy Source and State 1980-2017")
print()
#while trying to specify index, if it is just a number, it says invalid.
#Hence, add a prefix to the 'year' column
df.columns = ['Y_' + str(col) for col in df.columns]
#while adding prefix to 'year' column, other columns also get prefixed, remove their prefixes
df.rename(columns = {'Y_Energy_Source':'Energy_Source'}, inplace = True) 
df.rename(columns = {'Y_State':'State'}, inplace = True) 
#get the name of the columns and store them in a list
dfCols = list(df.columns)
#the first two rows of the list are not needed
dfCols = dfCols[2:]
#Step 1
#total consumption
#get the total consumption for every year (this includes consumption of petroleum, coal, natural gas and electricity)
df_1 = df.sum(axis=0)
#reset the index
df_1 = df_1.to_frame().reset_index()
#take the relevant values by removing the first two rows of the dataframe
df_1 = df_1.iloc[2:]
#rename the columns
df_1.columns = ['Year', 'Total_Consumption']
#take the relevant values by removing the first two rows of the dataframe
df_1.Year = df_1.Year.str[2:]
#specify the size of the plot and plot it
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df_1.Year, df_1.Total_Consumption, 'b')
#specifies the background color of the graph
#ax.set_facecolor('xkcd:light yellow')
ax.set_title('Year v/s Energy-Related Carbon Dioxide Emissions')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Total U.S. Energy-Related Carbon Dioxide Emissions')  #the naming for y-axis
plt.xticks(rotation=45) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#every year every resource
#the total consumption in every year by every resource is calculated
df_2 = df.groupby('Energy_Source')[dfCols].sum()
#every resource's consumption is stored in a list for plotting purpose
coalConsumption = df_2.values.tolist()[0]
electricityConsumption = df_2.values.tolist()[1]
naturalGasConsumption = df_2.values.tolist()[2]
petroleumConsumption = df_2.values.tolist()[3]

#all the energy source's consuption is shown on a single graph
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_1.Year, coalConsumption, label = 'Coal Consumed')
ax.plot(df_1.Year, electricityConsumption, label = 'Electricity Consumed')
ax.plot(df_1.Year, naturalGasConsumption, label = 'Natural gas Consumed')
ax.plot(df_1.Year, petroleumConsumption, label = 'Petroleum Consumed')

legend = ax.legend(loc='upper left', shadow=True, fontsize='large')

legend.get_frame().set_facecolor('w')
#ax.set_facecolor('xkcd:light white')
ax.set_title('Energy-Related Carbon Dioxide Emissions by Energy Source')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Total U.S. Energy-Related Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=45) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#get the energy resource that has consumed maximum in every year
#first get the resources in a 'series'
resourceName = df_2.idxmax(axis=0)
#get the consumed values in another 'series'
maxValues = df_2.max(axis=0)
#concatenate both the series
highestResource = pd.concat([resourceName, maxValues], axis=1).reset_index()
#rename the columns
highestResource.columns = ['Year', 'Energy_Resource', 'Consumption']
#remove irrelevant rows
highestResource.Year = highestResource.Year.str[2:]
#for plotting purpose, concatenate two rows
xAxis = highestResource.Energy_Resource + '-' +highestResource.Year
#plot the data
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xAxis, highestResource.Consumption)
legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('w')
#ax.set_facecolor('xkcd:light pink')
ax.set_title('Highest Energy-Related Carbon Dioxide Emissions - Petroleum')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Total U.S. Energy-Related Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#get the energy resource that has consumed minimum in every year
#first get the resources in a 'series'
resourceName = df_2.idxmin(axis=0)
#get the consumed values in another 'series'
minValues = df_2.min(axis=0)
#concatenate both the series
lowestResource = pd.concat([resourceName, minValues], axis=1).reset_index()
#rename the columns
lowestResource.columns = ['Year', 'Energy_Resource', 'Consumption']
#remove irrelevant rows
lowestResource.Year = lowestResource.Year.str[2:]
#for plotting purpose, concatenate two rows
xAxis = lowestResource.Energy_Resource + '-' +lowestResource.Year
#plot the data
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xAxis, lowestResource.Consumption)
legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('w')
ax.set_facecolor('xkcd:white')
ax.set_title('Lowest Energy-Related Carbon Dioxide Emissions - Natural Gas and Coal')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Total U.S. Energy-Related Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=45, ha = 'right') #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#showing year that consumed highest energy and lowest energy together
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(highestResource.Year, maxValues, label = 'Highest Amount of Energy-Related Carbon Dioxide Emissions')
ax.plot(highestResource.Year, minValues, label = 'Lowest Amount of Energy-Related Carbon Dioxide Emissions')
legend = ax.legend(loc='upper right', shadow=True, fontsize='large',
                   bbox_to_anchor=(0.5, -0.1))
legend.get_frame().set_facecolor('w')
ax.set_facecolor('xkcd:white')
ax.set_title('Highest and Lowest Energy-Related Carbon Dioxide Emissions')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Energy consumption')  #the namong for y-axis
plt.xticks(rotation=45) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#showing pie chart of energy consumed by every source over all the years
#sum of energy consumed is stored in 'series'
df_3 = df_2.sum(axis=1)
#it is converted to a dataframe
df_3 = df_3.to_frame().reset_index()
#the columns are renamed
df_3.columns = ['Energy_Source', 'Consumption']
#defining colors of the pie chart
colors = ["#E13F29", "#AE5552", "#CB5C3B", "#EB8076"]

# Create a pie chart
plt.pie(df_3['Consumption'],labels=df_3['Energy_Source'],
    shadow=False,
    colors=colors,
    # with one slide exploded out
    explode=(0, 0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )

# View the plot drop above
plt.axis('equal')
# View the plot
plt.tight_layout()
plt.show()

#the same pie chart is shown as a graph as well
#size of the graph is specified
fig, ax = plt.subplots(figsize=(12, 8))
#the x-axis values and y-axis values are specified
ax.bar(df_3.Energy_Source, df_3.Consumption)
#ax.set_facecolor('xkcd:light pink')
ax.set_title('Total U.S. Energy-Related Carbon Dioxide Emissions by Energy Source')
plt.xlabel('Energy Source') #the naming for x axis
plt.ylabel('Total U.S. Energy-Related Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=45) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()


#The data is divided into two parts- one for training, other for testing. 
#training data should be more for the predictions to be more accurate.
#we have taken 36 values out of 38 values as training data
#the 'trainingSet_data' is a list with 'Consumption' values which are mapped to values of the 'Year' column
trainingSet_data = highestResource.Year[:38]
yearVals = []
for i in range(2018, 2030,1):
    yearVals.append(i)
validateSet_data = yearVals
#for traiing purpose
targetSet_Coal = coalConsumption[:38]
targetSet_Electricity = electricityConsumption[:38]
targetSet_NaturalGas = naturalGasConsumption[:38]
targetSet_Petroleum = petroleumConsumption[:38]
#the remaining 2 values belong to test dataset
#validateSet_Coal = coalConsumption[36:]
#validateSet_Electricity = electricityConsumption[36:]
#validateSet_NaturalGas = naturalGasConsumption[36:]
#validateSet_Petroleum = petroleumConsumption[36:]
#The target values are those that are mapped to 'Year' column 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
coalList=[]
electricityList = []
naturalGasList = []
petroleumList = []   
def my_fun(training, resourcetype):
    print("Prediction Values for " +str(resourcetype) + " source")
    # Create linear regression object
    regr = linear_model.LinearRegression()
    #reshaping is done so that there are no discrepancies while predicting values
    X = np.array(trainingSet_data).reshape(-1, 1)
    y = np.array(training)        
    #the values are fit to the model using the 'fit' function
    regr.fit(X, y) 
    # Train the model using the training sets
    X_new = np.array(validateSet_data).reshape(-1, 1)
    #the prediction is done using the 'predict' function
    y_pred = regr.predict(X_new)
    #it is converted to int type from array for displaying purposes
    y_pred_int = y_pred.astype('int64')
    # The coefficients are displayed
    print()
    print('Coefficients: \n', regr.coef_)
    # The mean squared error is displayed
#    print('Mean squared error: %.2f'% mean_squared_error(validateSet_target, y_pred))
#    print()
#    print("The actual values of "+ str(resourcetype) +" carbon emissions for " )
#    print(str(validateSet_data.iloc[0]) + " is " + str(validateSet_target[0]) + " units")
#    print()
#    print("The actual values of "+ str(resourcetype) +" carbon emissions " )
#    print(str(validateSet_data.iloc[1]) + " is " + str(validateSet_target[1]) + " units")
#    print()
    for i in range(0,12):
        print("The predicted values of " + str(resourcetype) + " carbon emissions ")
        print(str(validateSet_data[i]) + " is " + str(y_pred_int[i]) + " units")
        print()
#        print("The predicted values of "+ str(resourcetype) +" carbon emissions ")
#        print(str(validateSet_data[1]) + " is " + str(y_pred_int[1]) + " units")
#        print()
    xVals = validateSet_data
    predictedVals_y = y_pred_int.tolist() 
    #    actualVals_y = validateSet_target
    fig, ax = plt.subplots(figsize=(12, 8))
    #    ax.plot(xVals, actualVals_y, label = 'Actual Energy-Related Carbon Dioxide Emissions for' + str(resourcetype))
    ax.scatter(xVals, predictedVals_y, label = 'Predicted Energy-Related Carbon Dioxide Emissions for' + str(resourcetype))
    legend.get_frame().set_facecolor('w')
        #ax.set_facecolor('xkcd:light pink')
    ax.set_title('Predicted Energy-Related Carbon Dioxide Emissions')
    plt.xlabel('Year') #the naming for x axis
    plt.ylabel(str(resourcetype) + ' Carbon Dioxide Emissions')  #the namong for y-axis
        #plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
    ax.ticklabel_format(style = 'plain', axis = 'y')
    plt.show()
    
    if resourcetype == 'Coal':
        coalList.append(predictedVals_y)
    if resourcetype == 'Electricity':
        electricityList.append(predictedVals_y)
    if resourcetype == 'Natural Gas':
        naturalGasList.append(predictedVals_y)
    if resourcetype == 'Petroleum':
        petroleumList.append(predictedVals_y)


my_fun(targetSet_Coal, 'Coal')
my_fun(targetSet_Electricity, 'Electricity')
my_fun(targetSet_NaturalGas, 'Natural Gas')
my_fun(targetSet_Petroleum, 'Petroleum')

import itertools

coalList = list(itertools.chain(*coalList))
electricityList = list(itertools.chain(*electricityList))
naturalGasList = list(itertools.chain(*naturalGasList))
petroleumList = list(itertools.chain(*petroleumList))
    

#as a line graph
print("Plotting them all on the same graph")
fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(year, consumption)
#ax.set_facecolor('xkcd:light pink')
ax.plot(yearVals, coalList, label = 'Coal carbon emission')
ax.plot(yearVals, electricityList, label = 'Electricity carbon emission')
ax.plot(yearVals, naturalGasList, label = 'Natural Gas carbon emission')
ax.plot(yearVals, petroleumList, label = 'Petroleum carbon emission')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large',
                   bbox_to_anchor=(0.5, -0.1))
legend.get_frame().set_facecolor('w')
ax.set_title('Predicted Energy-Related Carbon Dioxide Emissions')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=90) #so that the names of states are readable and don't overlap
ax.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

#as a scatter plot
print("Plotting them all on the same graph as scatter plots")
fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(year, consumption)
#ax.set_facecolor('xkcd:light pink')
ax.scatter(yearVals, coalList, label = 'Coal carbon emission')
ax.scatter(yearVals, electricityList, label = 'Electricity carbon emission')
ax.scatter(yearVals, naturalGasList, label = 'Natural Gas carbon emission')
ax.scatter(yearVals, petroleumList, label = 'Petroleum carbon emission')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large',
                   bbox_to_anchor=(0.5, -0.1))
legend.get_frame().set_facecolor('w')
ax.set_title('Predicted Energy-Related Carbon Dioxide Emissions')
plt.xlabel('Year') #the naming for x axis
plt.ylabel('Carbon Dioxide Emissions')  #the namong for y-axis
plt.xticks(rotation=90) #so that the names of states are readable and don't overlap

ax.ticklabel_format(style = 'plain', axis = 'y')
#ax.set_yscale('log')

plt.show()

