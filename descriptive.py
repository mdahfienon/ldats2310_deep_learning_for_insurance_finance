# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:20:20 2023

@author: MATHIAS
"""

######################################################
#                                                    #
#                EXPLORATORY ANALYSIS                #
#                                                    #
######################################################


#%% libraries

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% reading data

dt1 = pd.read_excel("Car Insurance Claim.xlsx", index_col="ID")

#%% verifiing the data type and some usual verification 

dt1.dtypes

#%% columns 

dt1.columns

#%% missing values checking

dt1.isnull().sum() # no missing value

#%%
######################################################
#                                                    #
#           DESCRIPTIVE STATITISCS                   #
#                                                    #
######################################################

#%% Home Children
cm = 1/2.54


dt1["Home Children"] = dt1["Home Children"].astype("category")

home_children = dt1["Home Children"].value_counts()/len(dt1["Home Children"])


plt.subplots(figsize=(20*cm, 12*cm))


plt.subplot(1,2,1)

plt.bar(home_children.index, home_children, 
        color = ["lightblue", "lightgreen", "olive", "c", "m"] )
plt.title("Number of Children at home for policyholder")
plt.xlabel("Number of children")
plt.ylabel("prop")


plt.subplot(1,2,2)


sns.countplot(dt1, x="Home Children", hue="Claims Flag (Crash)" )
plt.title("Number of Children at home for policyholder \n Claims Flag")


plt.tight_layout()
plt.show()

#%% Years on Job


plt.subplots(figsize=(20*cm, 12*cm))


plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Years on Job", 
             stat="percent", bins=20, kde=True)
plt.suptitle("Years on Job of policyholder")


plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Years on Job")


plt.tight_layout()
plt.show()


#%% Age

plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Age", stat="percent", bins=20, kde=True, color="c")
plt.suptitle("Age of policyholder")
#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Age", color="c")

# plt.subplot(1,3,3)
# sns.catplot(data=dt1, x = "Years on Job", y="Claims Flag (Crash)")

plt.tight_layout()
plt.show()


#%% Travel Time

plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Travel Time", stat="percent", bins=20, kde=True, color="c")
plt.suptitle("Travel Time to work of policyholder")
#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Travel Time", color="c")

# plt.subplot(1,3,3)
# sns.catplot(data=dt1, x = "Years on Job", y="Claims Flag (Crash)")

plt.tight_layout()
plt.show()


#%% Time In Force

plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Time In Force", stat="percent", bins=25, kde=True, color="cyan")
plt.suptitle("Time In Force of policyholder")
#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Time In Force", color="cyan")

# plt.subplot(1,3,3)
# sns.catplot(data=dt1, x = "Years on Job", y="Claims Flag (Crash)")

plt.tight_layout()
plt.show()


#%% Total Claims (5 Years)

plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Total Claims (5 Years)", stat="percent", bins=15, kde=True, color="cyan")
plt.suptitle("Claims severity")
#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Total Claims (5 Years)", color="cyan")

# plt.subplot(1,3,3)
# sns.catplot(data=dt1, x = "Years on Job", y="Claims Flag (Crash)")

plt.tight_layout()
plt.show()

#%% Income 

dt1["Income Norma"] = dt1["Income"]/10000


# plt.hist(dt1["Years on Job"], density=1, bins = list(range(0,100,3)) )
cm = 1/2.54  # centimeters in inches
plt.subplots(figsize=(20*cm, 12*cm))


plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Income Norma", color="c",
             stat="percent", bins=20, kde=True)
plt.xlabel("Income (X10000)")

#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Income Norma", color="c")
plt.ylabel("Income (X10000)")

plt.suptitle("Income distribution of policyholder")
plt.tight_layout()
plt.show()


#%% Single Parent?

dt1["Single Parent?"] = dt1["Single Parent?"].astype("category")

single_parent = dt1["Single Parent?"].value_counts()/len(dt1["Single Parent?"])

plt.subplots(figsize=(28*cm, 30*cm))


plt.subplot(3,2,1)
plt.bar(single_parent.index, single_parent, color = ["c", "m"] )
plt.title("Single Parent?")
# plt.xlabel("Number of children")
plt.ylabel("prop")
#plt.show()


# Marital Status

dt1["Marital Status"] = dt1["Marital Status"].astype("category")

marital_status = dt1["Marital Status"].value_counts()/len(dt1["Marital Status"])
plt.subplot(3,2,2)
plt.bar(marital_status.index, marital_status, color = ["c", "m"] )
plt.title("Marital Status")
# plt.xlabel("Number of children")
plt.ylabel("prop")
#plt.show()

#  marital status vs single parent
plt.subplot(3,2,3)
sns.countplot(dt1, x="Marital Status", hue="Single Parent?")
plt.title("Marital Status vs Single Parent?")

plt.subplot(3,2,4)
sns.countplot(dt1, x="Marital Status", hue="Claims Flag (Crash)")
plt.title("Marital Status of policyholder \n vs Claims Flag")


plt.subplot(3,2,5)
sns.countplot(dt1, x="Single Parent?", hue="Claims Flag (Crash)" )
plt.title("Single Parent? vs \n Claims Flag")

plt.tight_layout()
plt.show()

#%% Home Value


dt1["Home Value Norma"] = dt1["Home Value"]/10000

plt.subplots(figsize=(20*cm, 12*cm))


plt.subplot(1,2,1)
sns.histplot(data=dt1, x = "Home Value Norma", color="m",
             stat="percent", bins=20, kde=True)
plt.xlabel("Home Value (x10000)")
plt.suptitle("Home value of policyholder ")
#plt.show()

plt.subplot(1,2,2)
sns.boxplot(data = dt1, y="Home Value Norma", color="m")
plt.ylabel("Home Value (x10000)")

plt.tight_layout()
plt.show()



#%% Gender

plt.subplots(figsize=(20*cm, 12*cm))


plt.subplot(1,2,1)

dt1["Gender"] = dt1["Gender"].astype("category")

gender = dt1["Gender"].value_counts()/len(dt1["Gender"])

plt.bar(gender.index, gender, color = ["c", "m"] )
plt.title("Gender")
# plt.xlabel("Number of children")
plt.ylabel("prop")


plt.subplot(1,2,2)


sns.countplot(dt1, x="Gender", hue="Claims Flag (Crash)" )
plt.title("Gender of policyholder vs Claims Flag")


plt.tight_layout()
plt.show()

#%% Vehicule Points


dt1["Vehicle Points"] = dt1["Vehicle Points"].astype("category")

vehicule_points = dt1["Vehicle Points"].value_counts()/len(dt1["Vehicle Points"])


plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)
plt.bar(vehicule_points.index, vehicule_points, color = ["c", "m", "olive", "lightblue", "orange", "brown", "cyan", "purple"] )
plt.title("Vehicle Points")
# plt.xlabel("Number of children")
plt.ylabel("prop")


plt.subplot(1,2,2)
sns.countplot(dt1, x="Vehicle Points", hue="Claims Flag (Crash)" )
plt.title("Vehicle Points of policyholder vs Claims Flag")


plt.tight_layout()
plt.show()

# %% Education

dt1["Education"] = dt1["Education"].astype("category")

education = dt1["Education"].value_counts()/len(dt1["Education"])

plt.subplots(figsize=(20*cm, 12*cm))

plt.subplot(1,2,1)

plt.bar(education.index, education, color = ["c", "m", "olive", "lightblue"] )
plt.title("Education level")
# plt.xlabel("Number of children")
plt.ylabel("prop")
plt.xticks(rotation = 25)

plt.subplot(1,2,2)

sns.countplot(dt1, x="Education", hue="Claims Flag (Crash)" )
plt.title("Level of Education of policyholder vs Claims Flag")
plt.xticks(rotation = 25)

plt.tight_layout()
plt.show()

# %% Occupation

plt.subplots(figsize=(20*cm, 12*cm))
dt1["Occupation"] = dt1["Occupation"].astype("category")

occupation = dt1["Occupation"].value_counts()/len(dt1["Occupation"])

plt.subplot(1,2,1)
plt.bar(occupation.index, occupation, 
        color = ["c", "m", "olive", "lightblue", "orange", "brown", "cyan", "purple"])
plt.title("Occupation of policyholder")
# plt.xlabel("Number of children")
plt.xticks(rotation = 50)
plt.ylabel("prop")

plt.subplot(1,2,2)

sns.countplot(dt1, x="Occupation", hue="Claims Flag (Crash)" )
plt.title("Occupation of policyholder vs Claims Flag")
plt.xticks(rotation = 50)

plt.tight_layout()
plt.show()


# %% Car Use

plt.subplots(figsize=(20*cm, 12*cm))

dt1["Car Use"] = dt1["Car Use"].astype("category")

car_use = dt1["Car Use"].value_counts()/len(dt1["Car Use"])

plt.subplot(1,2,1)
plt.bar(car_use.index, car_use, 
        color = ["c", "m"])
plt.title("policyholder Car's Utility ")
# plt.xlabel("Number of children")
# plt.xticks(rotation = 25)
plt.ylabel("prop")

plt.subplot(1,2,2)

sns.countplot(dt1, x="Car Use", hue="Claims Flag (Crash)" )
plt.title("Car Utility vs Claims Flag")
plt.xlabel("Car Utility")
#plt.xticks(rotation = 25)

plt.tight_layout()
plt.show()


# %% Car Type

plt.subplots(figsize=(20*cm, 12*cm))
dt1["Car Type"] = dt1["Car Type"].astype("category")

car_type = dt1["Car Type"].value_counts()/len(dt1["Car Type"])

plt.subplot(1,2,1)
plt.bar(car_type.index, car_type, 
        color = ["c", "m", "olive", "lightblue", "orange", "brown", "cyan", "purple"]) 
plt.title("Car Type of policyholder ")
# plt.xlabel("Number of children")
plt.xticks(rotation = 50)
plt.ylabel("prop")

plt.subplot(1,2,2)

sns.countplot(dt1, x="Car Type", hue="Claims Flag (Crash)" )
plt.title("Car Type vs Claims Flag")
plt.xlabel("Car Type")
plt.xticks(rotation = 50)

plt.tight_layout()
plt.show()


# %% City Population

plt.subplots(figsize=(20*cm, 12*cm))
dt1["City Population"] = dt1["City Population"].astype("category")

city_population = dt1["City Population"].value_counts()/len(dt1["City Population"])

plt.subplot(1,2,1)
plt.bar(city_population.index, city_population, 
        color = ["c", "m", "olive", "lightblue", "orange", "brown", "cyan", "purple"]) 
plt.title("City Population  ")
# plt.xlabel("Number of children")
# plt.xticks(rotation = 50)
plt.ylabel("prop")

plt.subplot(1,2,2)

sns.countplot(dt1, x="City Population", hue="Claims Flag (Crash)" )
plt.title("City Population vs Claims Flag")
plt.xlabel("City Population")
# plt.xticks(rotation = 50)

plt.tight_layout()
plt.show()


#%% quick clearing of variables space

del car_type, car_use, city_population, cm, education, gender, home_children, marital_status, occupation, single_parent, vehicule_points


