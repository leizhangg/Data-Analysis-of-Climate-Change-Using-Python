
# 9650 Final Group Project

## Team Memembers: 
Shilpa Paidighantom, Carmen Ruan, Mingma Sherpa, 
Alexsandra Korzep, Katarzyna Borkowska, Lei Zhang
## Topic: Climate Change
---
**Problem Statement:**\
In this project, we are trying to answer the following questions:
* What are the key influencing factors of climate change in United States.
* How Meat Consumption, Food wastage are contributing to it?
* What will be the trend in next years?

**Objectives:**
* Perform factor analysis (PCA) to create climate change metric. Alternatively, climate change metric can be defined on one variable alone (such as daily average temperature)
* Build multivariate regression model to understand the drivers of climate change 
* Predict the impact of food industry on climate change and provide recommendations by states/clusters of states to reduce or control climate change 

**Table of Content**
1. [Introduction](#Introduction)
2. [Data Collection](#Data-Collection)
3. [Data Processing](#Data-Processing)
4. [Data Cleaning](#Data-Cleaning)
5. [Data Analysis](#Data-Analysis)
6. [Related Reference](#Related-Reference)

#### Introduction
Climate change has been occurring due to several reasons. Some variables we are analyzing for this project are CO2 emissions, electricity emissions, population, meat consumption, etc. As we analyze these variables, we want to find out which variables are more important to the contribution of climate change. There are several articles that support the reason for the variables we chose. These variables have been affecting the climate over time. CO2 emissions increase the temperature of the Earth. Human activities have influenced the change in the global water cycle, sea level and reduction in snow and ice. Meat consumption is also affecting the climate. As farms are needed to raise animals, people are cutting down trees and forests to clear the land and get more space. These farms produce huge amounts of greenhouse gases. Population increase has led to an increase in greenhouse gases. There will be an increase activity of burning fossil fuels and each person will contribute to producing carbon emissions.  Electricity emissions have also contributed to CO2 emissions as combustion of fossil fuels are needed to generate heat for vehicles. All of these variables have been causing complications in the Earth’s climate and it has been increasing significantly. We hope our analysis can justify the facts about climate change.


#### Data Collection
We gathered information on targeted variables identified as data requirements. All data is collected from various sources ranging from organizational databases to the information in web pages. View all the data [here](https://github.com/snowmeatball/9650_GroupProject/blob/main/Data/readme.md)

Targeted variables:
* CO2 emission
* Petroleum carbon emission
* Natural gas carbon emission
* Electricity carbon emission
* Coal emission
* Population
* Natural gas consumption
* Petroleum consumption
* Coal consumption
* Meat consumption

#### Data Processing
The data obtained, may not be structured and may contain irrelevant information. Hence, the collected data is required to be subjected to Data Processing and Data Cleaning.

* Extracted data from 1990 to 2000 to the dataset we need. Here is an example how the data is processed:
```
raw data
```
![1990raw](img/1990raw.png)
```
new data
```
![1990new](img/1990new.png)


* Methodology to calculate the contribution of the factors affecting the climate change:
1. Segmentation: All the states in the US will be segmented for each year from 2010 to 2020 based on the evidence metrics such as 
    * PRCP_max_yoy: Maximum precipitation on a particular day between 2010 to 2019 - Maximum precipitation on a particular day between 2000 to 2009
    * SNOW_max_yoy: Maximum snowfall on a particular day between 2010 to 2019 - Maximum snowfall on a particular day between 2000 to 2009
    * TMAX_max_yoy: Maximum temperature on a particular day between 2010 to 2019 - Maximum temperature on a particular day between 2000 to 2009
    * Maximum temperature on a particular day between 2020 to 2010 - Maximum temperature on a particular day between 2000 to 2009

   
   Results from each year will be analyzed to create clusters to define ‘Good’/‘Average’/‘Bad’ states based on climate change. Total of 10+ such  segmentations will be performed to further segment the states as 
    * States showing high climate change
    * States showing no climate change
    * States improving from ‘Bad’ to ‘Good’
    * States worsening from ‘Good’ to ‘Bad’

 ```
 Example data of year 2020
 ```
<p align="center" width="100%">
    <img width="50%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/exOf2020.png"> 
</p>

2. Regression model:  individual regression models are built on each of the clusters to understand the factors(targeted varibales) driving climate change. 
   The model is built from 1990 – 2019. Variables which did not have the time period. The metrics are calculated at state yearly level.
   The log transformation is taken on the variables to build the model. The models are often referred as log-log models as the transformation is taken on both independent and dependent variable. OLS estimation methodology is used, can improve the model efficiency with few more iterations. 


#### Data Cleaning
The processed and organized data may be incomplete, contain duplicates, or contain errors.
To prevent errors, incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data had been removed within a dataset. 

For example, the data below contains many columns with null values, which is incomplete and irrelevant to our topic, so those columns were removed.

<p align="center" width="100%">
    <img width="60%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/cleandata.png"> 
</p>


#### Data Analysis
Within the data, here is what we found:
```
Climate change is real. Both global average temperature(top graph)and North American average 
temperature has increased by 1 degree in recent 20 years. It is beyong natural change. 
```
<p align="center" width="100%">
    <img width="50%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/image1.png"> 
    <img width="50%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/NA_avg_tem.png"> 
</p>

```
Analyzing hottest years by month, we see years >= 2000 among top 10. 7 out of 10 hottest years 
for January is after the year 2020. This further shows how fast climate change is happening in 
the recent 20 years. 
```
<p align="center" width="100%">
    <img width="50%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/hot_year_by_month.png"> 
</p>

```
CO2 emission, petroleum emission,gas emission, coal emission and electricity emission has more
influence on climate changes than other factors such polulation, meat comsumption.
```
<p align="center" width="100%">
    <img width="70%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/factor.png"> 
</p>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/regression1.png"> 
    <img width="60%" src="https://github.com/snowmeatball/9650_GroupProject/blob/main/img/regression2.png"> 
</p>



#### Related Reference
* [Energy and the environment explained Greenhouse gases and the climate](https://www.eia.gov/energyexplained/energy-and-the-environment/greenhouse-gases-and-the-climate.php)
* [Eat less meat: UN climate-change report calls for change to human diet](https://www.nature.com/articles/d41586-019-02409-7)
* [Human Population Growth and Climate Change](https://www.biologicaldiversity.org/programs/population_and_sustainability/climate/)
* [Sources of Greenhouse Gas Emissions](https://www.epa.gov/ghgemissions/sources-greenhouse-gas-emissions)
* [Global Climate Change: What You Need to Know](https://www.nrdc.org/stories/global-climate-change-what-you-need-know)
* [Overview: Weather, Global Warming and Climate Change](https://climate.nasa.gov/resources/global-warming-vs-climate-change/)
* [Climate Change Datasets](https://github.com/adventuroussrv/Climate-Change-Datasets)
