#%%
from keras.src.utils import losses_utils
import tensorflow as tf
import numpy as np
import pandas as pd
from myPackage.my_module import *
# %%
df = pd.read_csv('dataForML.csv')
df = df.set_index(['Category', 'Index']).transpose()
df.head()
#%%
features = tf.constant(df.Bins.values)
targets = tf.constant(df.OptimalState.values)
# %%
from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(150, activation='sigmoid', input_shape = (100,)))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.compile('SGD', loss='MSE')
#%%
model.fit(features, targets, batch_size=10, epochs=50, validation_split = 0.2)

# %%
class custom_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(self)
    def call(self, y_true, y_pred):
        return     