#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from myPackage.my_module import *
#%%
data = load_samples('alldata')
#%%
data.Bins.head()
#%%
features = tf.constant(data.Bins)
target1 = tf.constant(data.Matrix.join(data.Measures.Distance))
target2 = tf.constant(data.OptimalState, dtype=tf.double)
#%%
def loss_function1(trues, parameters_preds):
    loss=0
    for true, parameters_pred in zip(trues, parameters_preds):
        true_np=true.numpy()
        parameters_pred_np = parameters_pred.numpy()    
        mat_pred = rho2(parameters_pred_np[0], parameters_pred_np[1])
        mat_true = true_np[0:16].reshape(4,4)
        dist_pred = matrix_fidelity(mat_pred, mat_true)
        loss+=max(0, dist_pred-true_np[-1])
    return loss/len(trues)
#%%
def loss_function2(y_true, y_pred):
    loss0 = tf.keras.losses.MeanSquaredError(y_true[:,0], y_pred[:,0])
    loss1 = tf.keras.losses.MeanSquaredError(y_true[:,1], y_pred[:,1])
    return np.sqrt(loss0**2 + loss1**2)

# %%
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Input
input = Input(shape=(100,))
conv = Conv1D(1, 5)(input)
hid1 = Dense(95, activation='sigmoid')(conv)
out = Dense(2, activation='sigmoid')(hid1)

model = Model(input, out)
model.compile('SGD', loss='MSE')
#%%
model.fit(features, target2, batch_size=1000, epochs=10, validation_split = 0.2)
# %%
model.predict(features[:100])  
# %%
density_matrix(rho2(0.1154, 0.80845)).histogram()
# %%
density_matrix(rho2(0.24484, 0.82018)).histogram()
# %%
plt.hist(data.OptimalState.Angle, bins=100)
# %%
plt.hist(data.OptimalState.Visibility, bins=100)
# %%
data.OptimalState.mean()
# %%
