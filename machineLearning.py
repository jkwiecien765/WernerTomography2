#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from myPackage.my_module import *
#%%
data = load_samples('alldata')
datac = load_samples('all_complex')
#%%
data.Bins.head()
#%%
features = tf.constant(data.Bins)
target1 = tf.constant(data.Matrix.join(data.Measures.Distance))
target2 = tf.constant(data.OptimalState, dtype=tf.double)
features_c = tf.constant(datac.Bins)
target1_c = tf.constant(datac.Matrix.join(datac.Measures.Distance))
target2_c = tf.constant(datac.OptimalState, dtype=tf.double)
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
#%%
centiles = [(np.percentile(datac.OptimalState.Angle, i), np.percentile(datac.OptimalState.Visibility, j)) for i in range(0, 100, 20) for j in range(0, 100, 20)]
segments = [(i,j) for i in np.linspace(0, max(datac.OptimalState.Angle),5) for j in np.linspace(0,1,5)]

#%%
def find_idx1(x):
    return next(i for i, val in enumerate(centiles) if val[0]>=x[0] and val[1]>=x[1])
def find_idx2(x):
    return next(i for i, val in enumerate(segments) if val[0]>=x[0] and val[1]>=x[1])

#%%

datac.Measures['AngleVisDiscrete'] = pd.Series([find_idx2(x) for x in zip(data.OptimalState.Angle, data.OptimalState.Visibility)], dtype=int)
#%%
target12_c = tf.constant(datac.Measures.AngleVisDiscrete)

# %%
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def model_ang_vis():
    input = Input(shape=(100,))
    hid1 = Dense(95, activation='sigmoid')(input)
    hid2 = Dense(95, activation='sigmoid')(input)
    out = Dense(25, activation='softmax')(hid2)
    loss = SparseCategoricalCrossentropy()
    return Model(input, out)
model = model_ang_vis()
model.compile('Adam', loss=loss)
#%%
model.fit(features_c, target12_c, batch_size=1000, epochs=10, validation_split = 0.2)
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
