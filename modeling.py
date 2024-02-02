# -*- coding: utf-8 -*-
"""
Created on Thus Nov 2 19:21:46 2023

@author: MATHIAS
"""
######################################################
#                                                    #
#                     MODELING                       #
#                                                    #
######################################################


#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns


#%% reading data

dt1 = pd.read_excel("Car Insurance Claim.xlsx", index_col="ID")

#%% preprocessing


######################################################
#                                                    #
#           PREPROCESSING TO DEEP LEARNING           #
#                                                    #
######################################################

from sklearn.model_selection import train_test_split

#%% proba of claims

def claims_proba(value):
    """
    little function to help create the outputs column    
    """
    if value == "No" : return 0
    else : return 1


dt1["y_i"] = dt1["Claims Flag (Crash)"].map(claims_proba)

dt1["y_i"] = dt1["y_i"].astype("category")

#%% categorical features 
dt1["Car Type"] = dt1["Car Type"].astype("category")
dt1["City Population"] = dt1["City Population"].astype("category")
dt1["Car Use"] = dt1["Car Use"].astype("category")
dt1["Car Use"] = dt1["Car Use"].astype("category")
dt1["Occupation"] = dt1["Occupation"].astype("category")
dt1["Education"] = dt1["Education"].astype("category")
dt1["Vehicle Points"] = dt1["Vehicle Points"].astype("category")
dt1["Gender"] = dt1["Gender"].astype("category")
dt1["Single Parent?"] = dt1["Single Parent?"].astype("category")
dt1["Marital Status"] = dt1["Marital Status"].astype("category")
dt1["Home Children"] = dt1["Home Children"].astype("category")

#%% creating dummies variables

dt2 = dt1.copy()


#%% defining categorical variables

dt2[["Red Car?", "License Revoked", "Claims Flag (Crash)"]] = dt2[["Red Car?", "License Revoked", "Claims Flag (Crash)"]].astype("category")

dt2.drop(inplace = True, columns = ["Claims Flag (Crash)", "Claims Amount", "DOB"]) # "Income Norma", "Home Value Norma", 

numerical_columns = list()

categorical_columns = list()

for i in dt2.dtypes.index:
    if dt2[str(i)].dtype == "category" : categorical_columns.append(str(i))
    else : numerical_columns.append(str(i))


del i

#%% preprocessing data for model1 : simple model

dt_model = dt2.copy()

for j in numerical_columns:
    dt_model[j] = (dt_model[j] - np.mean(dt_model[j])) / np.std(dt_model[j])

dt_model = pd.get_dummies(dt_model, drop_first=True, dtype = int, columns=categorical_columns[:-1])

dt_model1, dt_test = train_test_split(dt_model, test_size=0.2, random_state=102)

y_test = dt_test["y_i"]

X_test = dt_test.drop("y_i", axis=1, inplace=False)

y_i = dt_model1["y_i"]

dt_model1.drop("y_i", axis=1, inplace=True)

del j

#%% Architecture design


######################################################
#                                                    #
#           NEURAL NETWORK ARCHITECTURES             #
#                                                    #
######################################################

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#%% 1st Architecture : simple model

# splitting into train and val set
X_train_model_1, X_val_model_1, y_train_model_1, y_val_model_1 = train_test_split(dt_model1, y_i,
            test_size=0.3, random_state=101)


ncol_model_1    = X_train_model_1.shape[1]


inputs_model_1  = keras.Input(shape=[ncol_model_1], name="input_layers")
l       = layers.Dense(100, activation="tanh",
                       # kernel_regularizer=regularizers.L1(lam),
                       name="dense_1")(inputs_model_1)
l       = layers.Dense(60, activation="relu",
                       # kernel_regularizer=regularizers.L1(lam),
                       name="dense_2")(l)
l       = layers.Dense(30, activation="tanh", 
                       # kernel_regularizer=regularizers.L1(lam),
                       name="dense_3")(l)

l       = layers.Dense(9, activation="relu", 
                       # kernel_regularizer=regularizers.L1(lam),
                       name="dense_4")(l)

outputs_model_1 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

model_1 = keras.Model(inputs=inputs_model_1, outputs=outputs_model_1)

model_1.summary()

model_1.compile(
    optimizer=keras.optimizers.RMSprop(),  
    # Minimize loss:
    loss=keras.losses.BinaryCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.BinaryAccuracy()],
)

#calibration
print("Fit model on training data")
history_model_1 = model_1.fit(
    X_train_model_1, y_train_model_1,
    batch_size= 100,
    epochs    = 40,
    shuffle   = True,
    validation_data=(X_val_model_1, y_val_model_1)) 

#%% plot of loss vs accuracy

# plot loss
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_model_1.history['loss'], color='blue', label='train')
plt.plot(history_model_1.history['val_loss'], color='orange', label='val')
plt.grid()
plt.legend()


# plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_model_1.history['binary_accuracy'], color='blue', label='train')
plt.plot(history_model_1.history['val_binary_accuracy'], color='orange', label='val')
plt.grid()
plt.legend()
plt.tight_layout() #to avoid overlap of title
plt.show() 


#%% 
# predicted frequencies
y_pred_model_1      = model_1.predict(X_train_model_1)
y_pred_test_model_1 = model_1.predict(X_test)



# CE calculation
from sklearn.metrics import log_loss
CE_train_model_1   = log_loss(y_train_model_1, y_pred_model_1)
CE_test_model_1    = log_loss(y_test, y_pred_test_model_1)
print("CE_simple_model Training set : ", round(CE_train_model_1,4),
      "Deviance_simple_model Test set :",round(CE_test_model_1,4), 
      "freq_mean Test set :",round(np.mean(y_pred_test_model_1),4),
      sep="\n")

#%% 2nd Architecture : L1 regularized model

# copy of dt_model1
dt_model2 = dt_model1.copy()

# splitting into train and val set
X_train_model_2, X_val_model_2, y_train_model_2, y_val_model_2 = train_test_split(dt_model2, y_i,
            test_size=0.3, random_state=101)


ncol_model_2    = X_train_model_2.shape[1]

lam     = 1e-4

inputs_model_2  = keras.Input(shape=[ncol_model_2], name="input_layers")
l       = layers.Dense(100, activation="tanh",
                       kernel_regularizer=regularizers.L1(lam),
                       name="dense_1")(inputs_model_2)
l       = layers.Dense(60, activation="relu",
                       kernel_regularizer=regularizers.L1(lam),
                       name="dense_2")(l)
l       = layers.Dense(30, activation="tanh", 
                       kernel_regularizer=regularizers.L1(lam),
                       name="dense_3")(l)

l       = layers.Dense(9, activation="relu", 
                       kernel_regularizer=regularizers.L1(lam),
                       name="dense_4")(l)

outputs_model_2 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

model_2 = keras.Model(inputs=inputs_model_2, outputs=outputs_model_2)

model_2.summary()

#compilation
model_2.compile(
    optimizer=keras.optimizers.RMSprop(),  
    # Minimize loss:
    loss=keras.losses.BinaryCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.BinaryAccuracy()],
)

#calibration
print("Fit model on training data")
history_model_2 = model_2.fit(
    X_train_model_2, y_train_model_2,
    batch_size= 100,
    epochs    = 40,
    shuffle   = True,
    validation_data=(X_val_model_2, y_val_model_2)) 

#%% plot of loss vs accuracy

# plot loss
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_model_2.history['loss'], color='blue', label='train')
plt.plot(history_model_2.history['val_loss'], color='orange', label='val')
plt.grid()
plt.legend()


# plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_model_2.history['binary_accuracy'], color='blue', label='train')
plt.plot(history_model_2.history['val_binary_accuracy'], color='orange', label='val')
plt.grid()
plt.legend()
plt.tight_layout() #to avoid overlap of title
plt.show() 


#%% log_loss evaluation
# predicted frequencies
y_pred_model_2      = model_2.predict(X_train_model_2)
y_pred_test_model_2 = model_2.predict(X_test)



# CE calculation
from sklearn.metrics import log_loss
CE_train_model_2   = log_loss(y_train_model_2, y_pred_model_2)
CE_test_model_2    = log_loss(y_test, y_pred_test_model_2)
print("CE_model_2 Training set : ", round(CE_train_model_2,4),
      "Deviance_model_2 Test set :",round(CE_test_model_2,4), 
      "freq_mean Test set :",round(np.mean(y_pred_test_model_2),4),
      sep="\n")

#%% 3rd Architecture : L2 regularized model

# copy of dt_model1
dt_model3 = dt_model1.copy()

# splitting into train and val set
X_train_model_3, X_val_model_3, y_train_model_3, y_val_model_3 = train_test_split(dt_model3, y_i,
            test_size=0.3, random_state=101)

ncol_model_3    = X_train_model_3.shape[1]

lam     = 1e-4

inputs_model_3  = keras.Input(shape=[ncol_model_3], name="input_layers")
l       = layers.Dense(100, activation="tanh",
                       kernel_regularizer=regularizers.L2(lam),
                       name="dense_1")(inputs_model_3)
l       = layers.Dense(60, activation="relu",
                       kernel_regularizer=regularizers.L2(lam),
                       name="dense_2")(l)
l       = layers.Dense(30, activation="tanh", 
                       kernel_regularizer=regularizers.L2(lam),
                       name="dense_3")(l)

l       = layers.Dense(9, activation="relu", 
                       kernel_regularizer=regularizers.L2(lam),
                       name="dense_4")(l)

outputs_model_3 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

model_3 = keras.Model(inputs=inputs_model_3, outputs=outputs_model_3)

model_3.summary()

#compilation
model_3.compile(
    optimizer=keras.optimizers.RMSprop(),  
    # Minimize loss:
    loss=keras.losses.BinaryCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.BinaryAccuracy()])

#calibration
print("Fit model on training data")
history_model_3 = model_3.fit(
    X_train_model_3, y_train_model_3,
    batch_size= 100,
    epochs    = 40,
    shuffle   = True,
    validation_data=(X_val_model_3, y_val_model_3)) 

#%% plot of loss vs accuracy

# plot loss
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_model_3.history['loss'], color='blue', label='train')
plt.plot(history_model_3.history['val_loss'], color='orange', label='val')
plt.grid()
plt.legend()


# plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_model_3.history['binary_accuracy'], color='blue', label='train')
plt.plot(history_model_3.history['val_binary_accuracy'], color='orange', label='val')
plt.grid()
plt.legend()
plt.tight_layout() #to avoid overlap of title
plt.show() 


#%% log_loss evaluation

# predicted frequencies
y_pred_model_3      = model_3.predict(X_train_model_3)
y_pred_test_model_3 = model_3.predict(X_test)

# CE calculation
from sklearn.metrics import log_loss
CE_train_model_3   = log_loss(y_train_model_3, y_pred_model_3)
CE_test_model_3    = log_loss(y_test, y_pred_test_model_3)
print("CE_model_2 Training set : ", round(CE_train_model_3,4),
      "Deviance_model_2 Test set :",round(CE_test_model_3,4), 
      "freq_mean Test set :",round(np.mean(y_pred_test_model_3),4),
      sep="\n")



#%% 4th Architecture : L1L2 regularized model

# copy of dt_model1
dt_model4 = dt_model1.copy()


# splitting into train and val set
X_train_model_4, X_val_model_4, y_train_model_4, y_val_model_4 = train_test_split(dt_model4, y_i,
            test_size=0.3, random_state=101)


ncol_model_4    = X_train_model_4.shape[1]

lam     = 1e-4


inputs_model_4  = keras.Input(shape=[ncol_model_4], name="input_layers")
l       = layers.Dense(100, activation="tanh",
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_1")(inputs_model_4)
l       = layers.Dense(60, activation="relu",
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_2")(l)
l       = layers.Dense(30, activation="tanh", 
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_3")(l)

l       = layers.Dense(9, activation="relu", 
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_4")(l)

outputs_model_4 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

model_4 = keras.Model(inputs=inputs_model_4, outputs=outputs_model_4)

model_4.summary()

#compilation
model_4.compile(
    optimizer=keras.optimizers.RMSprop(),  
    # Minimize loss:
    loss=keras.losses.BinaryCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.BinaryAccuracy()],
)

#calibration
print("Fit model on training data")
history_model_4 = model_4.fit(
    X_train_model_4, y_train_model_4,
    batch_size= 100,
    epochs    = 40,
    shuffle   = True,
    validation_data=(X_val_model_4, y_val_model_4)) 

#%% plot of loss vs accuracy

# plot loss
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_model_4.history['loss'], color='blue', label='train')
plt.plot(history_model_4.history['val_loss'], color='orange', label='val')
plt.grid()
plt.legend()


# plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_model_4.history['binary_accuracy'], color='blue', label='train')
plt.plot(history_model_4.history['val_binary_accuracy'], color='orange', label='val')
plt.grid()
plt.legend()
plt.tight_layout() #to avoid overlap of title
plt.show() 

 
# model weights are saved in the current directory
# model.save_weights(filepath="modelLending.h5")
# model.load_weights("modelLending.h5")    


#%% log_loss evaluation
# predicted frequencies
y_pred_model_4      = model_4.predict(X_train_model_4)
y_pred_test_model_4 = model_4.predict(X_test)


# CE calculation
from sklearn.metrics import log_loss
CE_train_model_4   = log_loss(y_train_model_4, y_pred_model_4)
CE_test_model_4    = log_loss(y_test, y_pred_test_model_4)
print("CE_model_2 Training set : ", round(CE_train_model_4,4),
      "Deviance_model_2 Test set :",round(CE_test_model_4,4), 
      "freq_mean Test set :",round(np.mean(y_pred_test_model_4),4),
      sep="\n")


#%% 5th Architecture :Dropout model

# copy of dt_model1
dt_model5 = dt_model1.copy()


# splitting into train and val set
X_train_model_5, X_val_model_5, y_train_model_5, y_val_model_5 = train_test_split(dt_model5, y_i,
            test_size=0.3, random_state=101)


ncol_model_5    = X_train_model_5.shape[1]

lam     = 1e-4


inputs_model_5  = keras.Input(shape=[ncol_model_5], name="input_layers")
l       = layers.Dense(100, activation="tanh",
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_1")(inputs_model_5)
l       = layers.Dropout(.4)(l)
l       = layers.Dense(60, activation="relu",
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_2")(l)
l       = layers.Dropout(.4)(l)
l       = layers.Dense(30, activation="tanh", 
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_3")(l)
l       = layers.Dropout(.3)(l)
l       = layers.Dense(9, activation="relu", 
                       kernel_regularizer=regularizers.L1L2(lam),
                       name="dense_4")(l)

outputs_model_5 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

model_5 = keras.Model(inputs=inputs_model_5, outputs=outputs_model_5)

model_5.summary()

#compilation
model_5.compile(
    optimizer=keras.optimizers.RMSprop(),  
    # Minimize loss:
    loss=keras.losses.BinaryCrossentropy(),
    # Monitor metrics:
    metrics=[keras.metrics.BinaryAccuracy()],
)

#calibration
print("Fit model on training data")
history_model_5 = model_5.fit(
    X_train_model_5, y_train_model_5,
    batch_size= 250,
    epochs    = 100,
    shuffle   = True,
    validation_data=(X_val_model_5, y_val_model_5)) 

#%% plot of loss vs accuracy

# plot loss
#plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_model_5.history['loss'], color='blue', label='train')
plt.plot(history_model_5.history['val_loss'], color='orange', label='val')
plt.grid()
plt.legend()
plt.show() 
#%%
# plot accuracy
#plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_model_5.history['binary_accuracy'], color='blue', label='train')
plt.plot(history_model_5.history['val_binary_accuracy'], color='orange', label='val')
plt.grid()
plt.legend()
#plt.tight_layout() #to avoid overlap of title
plt.show() 


#%% log_loss evaluation

# predicted frequencies
y_pred_model_5      = model_5.predict(X_train_model_5)
y_pred_test_model_5 = model_5.predict(X_test)


# CE calculation
from sklearn.metrics import log_loss
CE_train_model_5   = log_loss(y_train_model_5, y_pred_model_5)
CE_test_model_5    = log_loss(y_test, y_pred_test_model_5)
print("CE_model_5 Training set : ", round(CE_train_model_5,4),
      "Deviance_model_5 Test set :",round(CE_test_model_5,4), 
      "freq_mean Test set :",round(np.mean(y_pred_test_model_5),4),
      sep="\n")

#%% model weights are saved in the current directory

model_5.save(filepath="model_5_best.h5") 
# model.load_weights("modelLending.h5")


#%% Predictions
######################################################
#                                                    #
#           PREDICTION WITH BEST MODEL               #
#                                                    #
######################################################


# predictions of my best model

dt_to_predict = dt_model.copy()

dt_to_predict_yi = dt_to_predict["y_i"]

dt_to_predict.drop("y_i", axis=1, inplace=True)


prediction =  model_5.predict(dt_to_predict)

dt_with_predict = dt2.copy()

print("prediction mean : ", np.mean(prediction))

dt_with_predict["y_i"] = dt_to_predict_yi

dt_with_predict["y_i"] = dt_with_predict["y_i"].astype("int")

dt_with_predict["prediction"] = prediction

#%% Comparison to real observed frequency

# function to obtained subgroup frequency
def avg_probabilities(subgroup, dt_with_predict = dt_with_predict ):
    
    """
    function to compute average proba per subgroup
    
    """
    result = dt_with_predict.groupby(subgroup).agg(
        Prob_real = ('y_i','mean'), Prob_pred = ('prediction','mean'))
    return result

#%% function to plot subgroup predictions and observed values 

# tool to plot observed and real claims frequencies
def plotFeatures(subgroup, data = dt_with_predict):
    """
    to make plot
    """
    data_used = avg_probabilities(subgroup, dt_with_predict = data)
    plt.plot(data_used.index, data_used.Prob_real,'go',label="Real")
    plt.plot(data_used.index, data_used.Prob_pred,'r*',label="NN")
    plt.xticks(rotation = 45)
    plt.grid()
    plt.title(subgroup)
    plt.legend()

#%%
cm = 1/2.54
plt.subplots(figsize=(45*cm, 36*cm))
ni = 1
for j in categorical_columns[:-1]:
    plt.subplot(4,3,ni)
    ni +=1
    plotFeatures(subgroup=j)

plt.tight_layout()
plt.show()

#%%
plt.subplots(figsize=(45*cm, 12*cm))

ni = 1

for j in categorical_columns[:3]:
    plt.subplot(1,3,ni)
    ni +=1
    plotFeatures(subgroup=j)
plt.tight_layout()
plt.show()

#%%

plt.subplots(figsize=(45*cm, 12*cm))

ni = 1

for j in categorical_columns[3:6]:
    plt.subplot(1,3,ni)
    ni +=1
    plotFeatures(subgroup=j)
plt.tight_layout()
plt.show()

#%%

plt.subplots(figsize=(45*cm, 12*cm))

ni = 1

for j in categorical_columns[6:9]:
    plt.subplot(1,3,ni)
    ni +=1
    plotFeatures(subgroup=j)
plt.tight_layout()
plt.show()

#%%

plt.subplots(figsize=(45*cm, 12*cm))

ni = 1

for j in categorical_columns[9:12]:
    plt.subplot(1,3,ni)
    ni +=1
    plotFeatures(subgroup=j)
plt.tight_layout()
plt.show()



#%% little clearing

del CE_test_model_5, CE_test_model_1, CE_test_model_2, CE_test_model_3, CE_test_model_4

del cm, CE_train_model_1, CE_train_model_2, CE_train_model_3, CE_train_model_4, CE_train_model_5

del ni, X_train_model_1, X_train_model_2, X_train_model_3, X_train_model_4, X_train_model_5

del X_val_model_1, X_val_model_2, X_val_model_3, X_val_model_4, X_val_model_5


#%%
######################################################
#                                                    #
#           MODEL INTERPRETABILITY                   #
#                                                    #
######################################################

#%%
###### PARTIAL DEPENDENCE and INDIVIDUAL CONDITIONAL EXPECTATION PLOTs



from keras.models import load_model
from scikeras.wrappers import KerasRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp


model = load_model("model_5_best.h5")


#%%

dt_model = dt2.copy()

for j in numerical_columns:
    dt_model[j] = (dt_model[j] - np.mean(dt_model[j])) / np.std(dt_model[j])

dt_model = pd.get_dummies(dt_model, drop_first=True, dtype=bool ,columns=categorical_columns[:-1])

dt_model1, dt_test = train_test_split(dt_model, test_size=0.2, random_state=102)

y_test = dt_test["y_i"]

X_test = dt_test.drop("y_i", axis=1, inplace=False)

y_i = dt_model1["y_i"]

dt_model1.drop("y_i", axis=1, inplace=True)

del j

#%%

dt_model5 = dt_model1.copy()

ncol_model_5    = dt_model5.shape[1]

#%%
def build_model():
    """
    model in pdp
    """
    lam     = 1e-4
    
    inputs_model_5  = keras.Input(shape=[ncol_model_5], name="input_layers")
    l       = layers.Dense(100, activation="tanh",
                           kernel_regularizer=regularizers.L1L2(lam),
                           name="dense_1")(inputs_model_5)
    l       = layers.Dropout(.4)(l)
    l       = layers.Dense(60, activation="relu",
                           kernel_regularizer=regularizers.L1L2(lam),
                           name="dense_2")(l)
    l       = layers.Dropout(.4)(l)
    l       = layers.Dense(30, activation="tanh", 
                           kernel_regularizer=regularizers.L1L2(lam),
                           name="dense_3")(l)
    l       = layers.Dropout(.3)(l)
    l       = layers.Dense(9, activation="relu", 
                           kernel_regularizer=regularizers.L1L2(lam),
                           name="dense_4")(l)

    outputs_model_5 = layers.Dense(1, activation="sigmoid", name="predictions")(l)

    model_5 = keras.Model(inputs=inputs_model_5, outputs=outputs_model_5)

    # model_5.summary()

    #compilation
    model_5.compile(
        optimizer=keras.optimizers.RMSprop(),  
        # Minimize loss:
        loss=keras.losses.BinaryCrossentropy(),
        # Monitor metrics:
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    return model_5


kr = KerasRegressor(build_model, epochs=100, batch_size=1000, verbose=0)
kr.fit(dt_model5, y_i, validation_split = 0.2)

features_plot = dt_model5.columns

#%% PDP and ICE

plt.close("all")


for i in range(11):
    pdp.from_estimator(kr, X_test, [i] , 
                       categorical_features=features_plot[11:],
                       method="brute", kind="both", centered=True) # features_plot
#%% PDP

for i in range(11):
    pdp.from_estimator(kr, X_test, [i] , 
                       categorical_features=features_plot[11:],
                       method="brute", kind="average", centered=True)   
    
#%% PDP for categorical features

for i in range(11, len(features_plot)):
    pdp.from_estimator(kr, X_test, [i] , 
                       categorical_features=features_plot[11:],
                       method="brute")
    
#%% PDP

for i in range(11, len(features_plot)):
    pdp.from_estimator(kr, X_test, [i] , 
                       categorical_features=features_plot[11:],
                       method="brute", kind="average")
    
#%%

pdp.from_estimator(kr, X_test, np.arange(11, 19) , 
                   categorical_features=features_plot[11:],
                   method="brute")

plt.tight_layout()
plt.show()


#%% lime 
######################################################
#                                                    #
#        FEATURES  IMPORTANCE FOR TWO POLICIES       #
#                                                    #
######################################################

#%%
# ################## LIME
from lime.lime_tabular import LimeTabularExplainer as lte

#%% explainer

# label_to_explain = list()


explainer = lte(np.asarray(X_test), mode="regression",
                feature_names=X_test.columns.values, 
                categorical_features = features_plot[11:],
                categorical_names=list(range(11, 52)))

explanation18 = explainer.explain_instance(np.asarray(X_test.iloc[155,]), 
                                         model.predict, 
                                         num_samples=2000)
explanation18.as_pyplot_figure()

explanation45 = explainer.explain_instance(np.asarray(X_test.iloc[45,]), 
                                         model.predict, 
                                         num_samples=2000)
explanation45.as_pyplot_figure()

#%%

id1 = dt_with_predict[dt2.index == 820425739]
# 155 = 820425739 

id2 = dt_with_predict[dt2.index == 834367523]
# 45 = 834367523

#%%

# ############## Shapley value
import shap

#%%
dt_model = dt2.copy()

for j in numerical_columns:
    dt_model[j] = (dt_model[j] - np.mean(dt_model[j])) / np.std(dt_model[j])

dt_model = pd.get_dummies(dt_model, drop_first=True, dtype = int, columns=categorical_columns[:-1])

dt_model1, dt_test = train_test_split(dt_model, test_size=0.2, random_state=102)

y_test = dt_test["y_i"]

X_test = dt_test.drop("y_i", axis=1, inplace=False)

y_i = dt_model1["y_i"]

dt_model1.drop("y_i", axis=1, inplace=True)

del j


#%%
explainer = shap.Explainer(model.predict, X_test)

#%%
shap_values = explainer(X_test)

#%%

shap.plots.waterfall(shap_values[155])

shap.plots.waterfall(shap_values[45])

#%%


