# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:22:31 2023

@author: MATHIAS
"""

#%% libraries

import numpy as np
import pandas as pd


#%% reading data

#dt = pd.read_csv('Car Insurance Claim.csv' )

dt1 = pd.read_excel("Car Insurance Claim.xlsx")

# %% data cleaning and check for NA values

dt1.isna().sum() # No NA values

#%% proba of claims

dt1["proba_of_claims"] = np.round(dt1["Claims Frequency (5 Years)"]/
                                  dt1["Claims Frequency (5 Years)"].sum(), 4)

#%% descriptive analysis


