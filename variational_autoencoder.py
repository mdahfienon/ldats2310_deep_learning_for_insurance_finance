# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:26:35 2023

@author: MATHIAS
"""

######################################################
#                                                    #
#           VARIATIONAL AUTOENCODERS                 #
#                                                    #
######################################################

#%% importing librairies

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import keras.backend as K

# from sklearn.model_selection import train_test_split

# Kmeans for post analysis
from sklearn.cluster import KMeans
from collections     import Counter


#%% importing the data

dt1 = pd.read_excel("Car Insurance Claim.xlsx", index_col="ID")

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


#%% data pre-processing

dt_var_encoder1 = dt2.copy()

labels_age = list()

for k in range(10):
    labels_age.append("age_" +str(k+1))

dt_var_encoder1["Age_cat"] = pd.cut(dt_var_encoder1["Age"],
                                   bins = 10, 
                                   labels=labels_age)

considered_features = ["Age_cat", "Single Parent?", 
                       "Marital Status", "Gender", "Education",
                       "Occupation", "Car Use", "Car Type", 
                       "City Population"]

dt_var_encoder = dt_var_encoder1[considered_features]

del dt_var_encoder1, labels_age, k

#%% get dummies of all variable 

dt_encoder = pd.get_dummies(dt_var_encoder, drop_first=False, dtype = float)

#%% VAE 1 


nc = dt_encoder.shape[1]

# number of latent variables
coding_size = 10

# definition of the encoder
encoder_inputs  = layers.Input(shape=(nc,))
z               = layers.Dense(20,activation="relu")(encoder_inputs)
z_mean          = layers.Dense(coding_size,name="z_mean")(z) 
z_log_var       = layers.Dense(coding_size,name="z_log_Var")(z)
variational_encoder = keras.Model(
    inputs=[encoder_inputs], outputs=[z_mean,z_log_var],name="encoder")

# definition of the decoder
decoder_inputs = layers.Input(shape=(coding_size,))
x              = layers.Dense(20,activation="relu")(decoder_inputs)
decoder_outputs= layers.Dense(nc, activation="sigmoid")(x)              
variational_decoder= keras.Model(
    inputs=[decoder_inputs],outputs=[decoder_outputs],name="decoder")

# Define the VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder,wgt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt         #weights for the MSE

    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        #bin_loss            = tf.losses.BinaryCrossentropy()
        #reconstruction_loss = bin_loss(inputs,outputs)
        #reconstruction_loss = tf.losses.MSE(inputs,outputs)
        reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt))
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss
                
# Instantiate the VAE model, we pass as argument the stdev of X
std_X = np.std(dt_encoder, axis=0)
vae   = VAE(variational_encoder, variational_decoder, std_X)
# Compile and fit
vae.compile(
    optimizer=keras.optimizers.RMSprop())
 

# dt_train, dt_test = train_test_split(dt_encoder, test_size=0.3, 
#                                      random_state=102)


#calibration
print("Fit model on training data")
history = vae.fit(
    dt_encoder,
    dt_encoder,
    batch_size=200,
    epochs    =120,
    shuffle   = True,
    # Validation of loss and metrics
    # at the end of each epoch:
    # validation_data=(dt_test, dt_test)
)


print("loss of this model = ", history.history['loss'][-1])

  
_,X_compressed,log_var, reconstruct = vae.predict(dt_encoder)

from scipy.stats import entropy

kl_divergences = []
for i in range(len(dt_encoder)):
    kl_divergences.append(entropy(dt_encoder.iloc[i], reconstruct[i]))

# Average KL divergence across all data points
average_kl_divergence = np.round(np.mean(kl_divergences),4)
print(f"Average KL Divergence: {average_kl_divergence}")


#%% VAE 2 


nc = dt_encoder.shape[1]

# number of latent variables
coding_size = 20

# definition of the encoder
encoder_inputs  = layers.Input(shape=(nc,))
z               = layers.Dense(150,activation="relu")(encoder_inputs)
z               = layers.Dense(100,activation="selu")(z)
z               = layers.Dense(50,activation="selu")(z)
z                = layers.Dense(10,activation="selu")(z)
z_mean          = layers.Dense(coding_size,name="z_mean")(z) 
z_log_var       = layers.Dense(coding_size,name="z_log_Var")(z)
variational_encoder = keras.Model(
    inputs=[encoder_inputs], outputs=[z_mean,z_log_var],name="encoder")

# definition of the decoder
decoder_inputs = layers.Input(shape=(coding_size,))
x              = layers.Dense(10,activation="relu")(decoder_inputs)
x              = layers.Dense(50,activation="selu")(x)
x              = layers.Dense(100,activation="selu")(x)
x              = layers.Dense(150,activation="selu")(x)
decoder_outputs= layers.Dense(nc, activation="sigmoid")(x)              
variational_decoder= keras.Model(
    inputs=[decoder_inputs],outputs=[decoder_outputs], name="decoder")

# Define the VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, wgt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt         #weights for the MSE

    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        #bin_loss            = tf.losses.BinaryCrossentropy()
        #reconstruction_loss = bin_loss(inputs,outputs)
        #reconstruction_loss = tf.losses.MSE(inputs,outputs)
        reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt))
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss
                
# Instantiate the VAE model, we pass as argument the stdev of X
std_X = np.std(dt_encoder, axis=0)
vae   = VAE(variational_encoder, variational_decoder, std_X)
# Compile and fit
vae.compile(
    optimizer=keras.optimizers.RMSprop())
 

# dt_train, dt_test = train_test_split(dt_encoder, test_size=0.3, 
#                                      random_state=102)


#calibration
print("Fit model on training data")
history = vae.fit(
    dt_encoder,
    dt_encoder,
    batch_size=200,
    epochs    =120,
    shuffle   = True,
    # Validation of loss and metrics
    # at the end of each epoch:
    # validation_data=(dt_test, dt_test)
)


print("loss of this model = ", history.history['loss'][-1])

  
_,X_compressed,log_var, reconstruct = vae.predict(dt_encoder)

from scipy.stats import entropy

kl_divergences = []
for i in range(len(dt_encoder)):
    kl_divergences.append(entropy(dt_encoder.iloc[i], reconstruct[i]))

# Average KL divergence across all data points
average_kl_divergence = np.round(np.mean(kl_divergences),4)
print(f"Average KL Divergence: {average_kl_divergence}")



#%% VAE 3
nc = dt_encoder.shape[1]

# number of latent variables
coding_size = 30

# definition of the encoder
encoder_inputs  = layers.Input(shape=(nc,))
z               = layers.Dense(150,activation="relu")(encoder_inputs)
z               = layers.Dense(100,activation="selu")(z)
z               = layers.Dense(50,activation="selu")(z)
z                = layers.Dense(10,activation="selu")(z)
z_mean          = layers.Dense(coding_size,name="z_mean")(z) 
z_log_var       = layers.Dense(coding_size,name="z_log_Var")(z)
variational_encoder = keras.Model(
    inputs=[encoder_inputs], outputs=[z_mean,z_log_var],name="encoder")

# definition of the decoder
decoder_inputs = layers.Input(shape=(coding_size,))
x              = layers.Dense(10,activation="relu")(decoder_inputs)
x              = layers.Dense(50,activation="selu")(x)
x              = layers.Dense(100,activation="selu")(x)
x              = layers.Dense(150,activation="selu")(x)
decoder_outputs= layers.Dense(nc, activation="sigmoid")(x)              
variational_decoder= keras.Model(
    inputs=[decoder_inputs],outputs=[decoder_outputs], name="decoder")

# Define the VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, wgt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt         #weights for the MSE

    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        #bin_loss            = tf.losses.BinaryCrossentropy()
        #reconstruction_loss = bin_loss(inputs,outputs)
        #reconstruction_loss = tf.losses.MSE(inputs,outputs)
        reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt))
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss
                
# Instantiate the VAE model, we pass as argument the stdev of X
std_X = np.std(dt_encoder, axis=0)
vae   = VAE(variational_encoder, variational_decoder, std_X)
# Compile and fit
vae.compile(
    optimizer=keras.optimizers.RMSprop())
 

# dt_train, dt_test = train_test_split(dt_encoder, test_size=0.3, 
#                                      random_state=102)


#calibration
print("Fit model on training data")
history = vae.fit(
    dt_encoder,
    dt_encoder,
    batch_size=200,
    epochs    =120,
    shuffle   = True,
    # Validation of loss and metrics
    # at the end of each epoch:
    # validation_data=(dt_test, dt_test)
)


print("loss of this model = ", history.history['loss'][-1])

  
_, X_compressed, log_var, reconstruct = vae.predict(dt_encoder)

from scipy.stats import entropy

kl_divergences = []
for i in range(len(dt_encoder)):
    kl_divergences.append(entropy(dt_encoder.iloc[i], reconstruct[i]))

# Average KL divergence across all data points
average_kl_divergence = np.round(np.mean(kl_divergences),4)
print(f"Average KL Divergence: {average_kl_divergence}")


#%% VAE 4
nc = dt_encoder.shape[1]

# number of latent variables
coding_size = 5

# definition of the encoder
encoder_inputs  = layers.Input(shape=(nc,))
z               = layers.Dense(20,activation="relu")(encoder_inputs)
z               = layers.Dense(100,activation="selu")(z)
z               = layers.Dense(50,activation="selu")(z)
z                = layers.Dense(10,activation="selu")(z)
z_mean          = layers.Dense(coding_size,name="z_mean")(z) 
z_log_var       = layers.Dense(coding_size,name="z_log_Var")(z)
variational_encoder = keras.Model(
    inputs=[encoder_inputs], outputs=[z_mean,z_log_var],name="encoder")

# definition of the decoder
decoder_inputs = layers.Input(shape=(coding_size,))
x              = layers.Dense(20,activation="relu")(decoder_inputs)
x              = layers.Dense(50,activation="selu")(x)
x              = layers.Dense(100,activation="selu")(x)
x              = layers.Dense(150,activation="selu")(x)
decoder_outputs= layers.Dense(nc, activation="sigmoid")(x)              
variational_decoder= keras.Model(
    inputs=[decoder_inputs],outputs=[decoder_outputs], name="decoder")

# Define the VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, wgt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt         #weights for the MSE

    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        #bin_loss            = tf.losses.BinaryCrossentropy()
        #reconstruction_loss = bin_loss(inputs,outputs)
        #reconstruction_loss = tf.losses.MSE(inputs,outputs)
        reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt))
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss
                
# Instantiate the VAE model, we pass as argument the stdev of X
std_X = np.std(dt_encoder, axis=0)
vae   = VAE(variational_encoder, variational_decoder, std_X)
# Compile and fit
vae.compile(
    optimizer=keras.optimizers.RMSprop())
 

# dt_train, dt_test = train_test_split(dt_encoder, test_size=0.3, 
#                                      random_state=102)


#calibration
print("Fit model on training data")
history = vae.fit(
    dt_encoder,
    dt_encoder,
    batch_size=200,
    epochs    =120,
    shuffle   = True,
    # Validation of loss and metrics
    # at the end of each epoch:
    # validation_data=(dt_test, dt_test)
)


print("loss of this model = ", history.history['loss'][-1])

  
_,X_compressed,log_var, reconstruct = vae.predict(dt_encoder)

from scipy.stats import entropy

kl_divergences = []
for i in range(len(dt_encoder)):
    kl_divergences.append(entropy(dt_encoder.iloc[i], reconstruct[i]))

# Average KL divergence across all data points
average_kl_divergence = np.round(np.mean(kl_divergences),4)
print(f"Average KL Divergence: {average_kl_divergence}")

#%% reconstruction quality


# Function that reconstructs a non-dummified dataset from the compressed
# information. We select the most likely feature.

def undummify(df, prefix_sep="_"):
    # cols2collapse, dictionary of columns to collapse
    cols2collapse = {
       item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    # new dataframe
    series_list   = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            # we select the modality with the maximum probability
            undummified = (df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col))
            # we add the new undummified column
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

# We reconstruct the dataset
# _,_,_,X_new_one_hot = vae.predict(dt_encoder)



reconstruted_columns = list()

for k in dt_encoder.columns:
    reconstruted_columns.append(k) # +"_reconstruct"
    
# Temporary dataset that serves as input of the function undummify
df        = pd.DataFrame(reconstruct, columns = reconstruted_columns, 
                         index = dt_var_encoder.index)

X_reconst = undummify(df, prefix_sep="_")

X_reconst = X_reconst.astype("category")

def quality_reconstruction(name1 = "Age_cat", name2 = "Age", rotat = 0):
    
    """
    plot to show the quality of the reconstruction
    
    """

    xlab = np.arange(1, len(dt_var_encoder[name1].value_counts().index)+1)
    
    x_name = list(dt_var_encoder[name1].value_counts().index)
    
    plt.bar(xlab-0.2, dt_var_encoder[name1].value_counts(), 0.4,  
            color = "c", label = "real data")
    plt.bar(xlab+0.2, X_reconst[name2].value_counts(), 0.4, 
            color = "lightblue", label = "reconstructed data")
    plt.xticks( ticks = xlab, rotation = rotat, labels = x_name)
    plt.legend()
    plt.title("Reconstruction quality for " + name2)
    plt.show()

#%% reconstruction plot age

quality_reconstruction(rotat= 30)

#%% reconstruction plot

for i in X_reconst.columns[1:]:
    if i == "Occupation" : quality_reconstruction(i, i, rotat= 30)
    else :  quality_reconstruction(i, i)



#%%
######################################################
#                                                    #
#                     K-MEANS                        #
#                                                    #
######################################################


#%% kmeans 10 
# n_clus = 10


n_clus = 10

# we run the K-means algorithm

km = KMeans(init="random",
            n_clusters= n_clus,
            n_init    = 20,
            max_iter  = 300,
            random_state=42)

#seed of the rnd number generator

km.fit(X_compressed )

# goodness of fit
print('K-means inertia {}'.format(km.inertia_))

# vector of clusters associated to each record 
X_clus    = km.labels_ 
# X_compress = variational_encoder.fit()


out_names = ['Cluster', "Age_cat", "Single Parent?", 
                       "Marital Status", "Gender", "Education",
                       "Occupation", "Car Use", "Car Type", 
                       "City Population", 'Frequency']

out_tab   = np.zeros(shape=(n_clus, len(out_names)))

out_tab   = pd.DataFrame(data=out_tab, columns = out_names)


for k in range(0, n_clus):
    idx  = (X_clus==k)
    freq = sum(dt2['y_i'][idx])/sum(dt1["y_i"]) # /sum(dt2['Duration'][idx])
    G = Counter(dt_var_encoder['Age_cat'][idx]).most_common(1)
    Z = Counter(dt_var_encoder['Single Parent?'][idx]).most_common(1)
    C = Counter(dt_var_encoder['Marital Status'][idx]).most_common(1)
    O = Counter(dt_var_encoder['Gender'][idx]).most_common(1)
    V = Counter(dt_var_encoder['Education'][idx]).most_common(1)
    V1 = Counter(dt_var_encoder['Occupation'][idx]).most_common(1)
    V2 = Counter(dt_var_encoder['Car Use'][idx]).most_common(1)
    V3 = Counter(dt_var_encoder['Car Type'][idx]).most_common(1)
    V4 = Counter(dt_var_encoder['City Population'][idx]).most_common(1)
    tmp = [k, G[0][0], Z[0][0], C[0][0],
              O[0][0], V[0][0], V1[0][0], V2[0][0], V3[0][0], 
              V4[0][0], round(freq, 2)]   
    out_tab.loc[k]=tmp          
# print and save data    
# print(out_tab)


from sklearn.metrics import log_loss

ypred           = out_tab['Frequency'][X_clus]
yobs            = dt2['y_i'] #data['Frequency']
mean_deviance   = log_loss(yobs , ypred)

# mean_poisson_deviance(yobs,ypred,sample_weight=data['Duration'])
# total_deviance  = mean_deviance *sum(Y['Duration'])  #*len(Y) error because weigthed mean
print("Deviance :", mean_deviance)


#%% kmeans 15 

n_clus = 15

# we run the K-means algorithm

km = KMeans(init="random",
            n_clusters= n_clus,
            n_init    = n_clus+20,
            max_iter  = 300,
            random_state=42)

#seed of the rnd number generator

km.fit(X_compressed )

# goodness of fit
print('K-means inertia {}'.format(km.inertia_))

# vector of clusters associated to each record 
X_clus    = km.labels_ 
# X_compress = variational_encoder.fit()


out_names = ['Cluster', "Age_cat", "Single Parent?", 
                       "Marital Status", "Gender", "Education",
                       "Occupation", "Car Use", "Car Type", 
                       "City Population", 'Frequency']

out_tab   = np.zeros(shape=(n_clus, len(out_names)))

out_tab   = pd.DataFrame(data=out_tab, columns = out_names)


for k in range(0, n_clus):
    idx  = (X_clus==k)
    freq = sum(dt2['y_i'][idx])/sum(dt1["y_i"]) # /sum(dt2['Duration'][idx])
    G = Counter(dt_var_encoder['Age_cat'][idx]).most_common(1)
    Z = Counter(dt_var_encoder['Single Parent?'][idx]).most_common(1)
    C = Counter(dt_var_encoder['Marital Status'][idx]).most_common(1)
    O = Counter(dt_var_encoder['Gender'][idx]).most_common(1)
    V = Counter(dt_var_encoder['Education'][idx]).most_common(1)
    V1 = Counter(dt_var_encoder['Occupation'][idx]).most_common(1)
    V2 = Counter(dt_var_encoder['Car Use'][idx]).most_common(1)
    V3 = Counter(dt_var_encoder['Car Type'][idx]).most_common(1)
    V4 = Counter(dt_var_encoder['City Population'][idx]).most_common(1)
    tmp = [k, G[0][0], Z[0][0], C[0][0],
              O[0][0], V[0][0], V1[0][0], V2[0][0], V3[0][0], 
              V4[0][0], round(freq, 2)]   
    out_tab.loc[k]=tmp          

# print and save data    
# print(out_tab)


from sklearn.metrics import log_loss
# from sklearn.metrics import mean_poisson_deviance

ypred           = out_tab['Frequency'][X_clus]
yobs            = dt2['y_i'] #data['Frequency']
mean_deviance   = log_loss(yobs , ypred)

# mean_poisson_deviance(yobs,ypred,sample_weight=data['Duration'])
# total_deviance  = mean_deviance *sum(Y['Duration'])  #*len(Y) error because weigthed mean
print("Deviance :", mean_deviance)

#%% kmeans 20 
# n_clus = 10


n_clus = 20

# we run the K-means algorithm

km = KMeans(init="random",
            n_clusters= n_clus,
            n_init    = n_clus+20,
            max_iter  = 300,
            random_state=42)

# seed of the rnd number generator

km.fit(X_compressed )

# goodness of fit
print('K-means inertia {}'.format(km.inertia_))

# vector of clusters associated to each record 
X_clus    = km.labels_ 
# X_compress = variational_encoder.fit()


out_names = ['Cluster', "Age_cat", "Single Parent?", 
                       "Marital Status", "Gender", "Education",
                       "Occupation", "Car Use", "Car Type", 
                       "City Population", 'Frequency']

out_tab   = np.zeros(shape=(n_clus, len(out_names)))

out_tab   = pd.DataFrame(data=out_tab, columns = out_names)


for k in range(0, n_clus):
    idx  = (X_clus==k)
    freq = sum(dt2['y_i'][idx])/sum(dt1["y_i"]) # /sum(dt2['Duration'][idx])
    G = Counter(dt_var_encoder['Age_cat'][idx]).most_common(1)
    Z = Counter(dt_var_encoder['Single Parent?'][idx]).most_common(1)
    C = Counter(dt_var_encoder['Marital Status'][idx]).most_common(1)
    O = Counter(dt_var_encoder['Gender'][idx]).most_common(1)
    V = Counter(dt_var_encoder['Education'][idx]).most_common(1)
    V1 = Counter(dt_var_encoder['Occupation'][idx]).most_common(1)
    V2 = Counter(dt_var_encoder['Car Use'][idx]).most_common(1)
    V3 = Counter(dt_var_encoder['Car Type'][idx]).most_common(1)
    V4 = Counter(dt_var_encoder['City Population'][idx]).most_common(1)
    tmp = [k, G[0][0], Z[0][0], C[0][0],
              O[0][0], V[0][0], V1[0][0], V2[0][0], V3[0][0], 
              V4[0][0], round(freq, 2)]   
    out_tab.loc[k]=tmp          

# print and save data    
# print(out_tab)


from sklearn.metrics import log_loss
# from sklearn.metrics import mean_poisson_deviance

ypred           = out_tab['Frequency'][X_clus]
yobs            = dt2['y_i'] #data['Frequency']
mean_deviance   = log_loss(yobs , ypred)

# mean_poisson_deviance(yobs,ypred,sample_weight=data['Duration'])
# total_deviance  = mean_deviance *sum(Y['Duration'])  #*len(Y) error because weigthed mean
print("Deviance :", mean_deviance)


#%% kmeans 5 

n_clus = 5

# we run the K-means algorithm

km = KMeans(init="random",
            n_clusters= n_clus,
            n_init    = n_clus+20,
            max_iter  = 300,
            random_state=42)

#seed of the rnd number generator

km.fit(X_compressed )

# goodness of fit
print('K-means inertia {}'.format(km.inertia_))

# vector of clusters associated to each record 
X_clus    = km.labels_ 
# X_compress = variational_encoder.fit()


out_names = ['Cluster', "Age_cat", "Single Parent?", 
                       "Marital Status", "Gender", "Education",
                       "Occupation", "Car Use", "Car Type", 
                       "City Population", 'Frequency']

out_tab   = np.zeros(shape=(n_clus, len(out_names)))

out_tab   = pd.DataFrame(data=out_tab, columns = out_names)


for k in range(0, n_clus):
    idx  = (X_clus==k)
    freq = sum(dt2['y_i'][idx])/sum(dt1["y_i"]) # /sum(dt2['Duration'][idx])
    G = Counter(dt_var_encoder['Age_cat'][idx]).most_common(1)
    Z = Counter(dt_var_encoder['Single Parent?'][idx]).most_common(1)
    C = Counter(dt_var_encoder['Marital Status'][idx]).most_common(1)
    O = Counter(dt_var_encoder['Gender'][idx]).most_common(1)
    V = Counter(dt_var_encoder['Education'][idx]).most_common(1)
    V1 = Counter(dt_var_encoder['Occupation'][idx]).most_common(1)
    V2 = Counter(dt_var_encoder['Car Use'][idx]).most_common(1)
    V3 = Counter(dt_var_encoder['Car Type'][idx]).most_common(1)
    V4 = Counter(dt_var_encoder['City Population'][idx]).most_common(1)
    tmp = [k, G[0][0], Z[0][0], C[0][0],
              O[0][0], V[0][0], V1[0][0], V2[0][0], V3[0][0], 
              V4[0][0], round(freq, 2)]   
    out_tab.loc[k]=tmp          

# print and save data    
# print(out_tab)


from sklearn.metrics import log_loss
# from sklearn.metrics import mean_poisson_deviance

ypred           = out_tab['Frequency'][X_clus]
yobs            = dt2['y_i'] #data['Frequency']
mean_deviance   = log_loss(yobs , ypred)

# mean_poisson_deviance(yobs,ypred,sample_weight=data['Duration'])
# total_deviance  = mean_deviance *sum(Y['Duration'])  #*len(Y) error because weigthed mean
print("Deviance :", mean_deviance)


#%% saving clusters
out_tab.to_csv("cluster5.csv")
