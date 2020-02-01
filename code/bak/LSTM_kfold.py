import sys
import glob
import os
import math
import pandas as pd
import numpy as np
import geopandas as gp

import matplotlib.pyplot as plt
import rsfuncs as rs

from scipy import stats
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from pandas.tseries.offsets import MonthEnd, SemiMonthEnd
from datetime import datetime, timedelta
from datetime import datetime as dt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Read data
states_file = gp.read_file("../shape/states.shp")
states = states_file[~states_file['STATE_ABBR'].isin(["AK", "HI", "PR", "VI"])]

site_file = gp.read_file("../shape/scan_sites.shp")
sites = site_file[~site_file['state'].isin(["AK", "HI", "PR", "VI"])]


# formate data 
df = pd.read_csv("../data/all_dat_f.csv")
df.rename(columns={ df.columns[0]: "site" , df.columns[1]:"date"}, inplace = True)
df.replace([np.inf, -np.inf], np.nan, inplace = True)
df = df.set_index(pd.to_datetime(df.date))
df['date'] = pd.to_datetime(df.date)

# Filter the data. 

# Drop days with 2day precip less than 1 mm
df = df[df.precip<1]

# Remove Urban Areas
df = df[df.lc_type != 2]
df = df[df.lc_type != 3]

# Remove sites with <10 datapoints
for i in df.site.unique():
    if len(df[df.site == i]) < 10:
        df = df[df.site != i]


# Calculate spectral indices
df['ndvi'] = (df.B5 - df.B4) / (df.B5 + df.B4)
df["ndmi"] = (df.B5 - df.B6) / (df.B5 + df.B6)
df["evi"] = 2.5*(df.B5 - df.B4) / (df.B5 + 6*df.B4- 7.5*df.B2 + 1)


# For the backscatter columns (df.vv, df.hv), delete any zeros, nans, deal with weird formatting, and calc the mean 

vv_eff = []

for i in df.vv:
    line = i.replace("[","")
    line = line.replace("]","")
    line = ' '.join(line.split())
    data = [float(i) for i in line.split(' ')]
    data = [i for i in data if i !=0.]
    vv_eff.append(np.nanmean(data))
    

hv_eff = []

for i in df.hv:
    if type(i) is float:
        hv_eff.append(np.nan)
    else:
        line = i.replace("[","")
        line = line.replace("]","")
        line = ' '.join(line.split())
        data = [float(i) for i in line.split(' ')]
        data = [i for i in data if i !=0.]
        hv_eff.append(np.nanmean(data))


df['vv'] = vv_eff
df['hv'] = hv_eff

# calc the 12 day means for each site: 
df = df.groupby(['site']).resample('12D').mean().fillna(np.nan).dropna()


def compute_lags(df, n=3):
    df['vv_t1'] = df['vv'].shift(1)
    df['hv_t1'] = df['hv'].shift(1)
    df['vv_t2'] = df['vv'].shift(2)
    df['hv_t2'] = df['hv'].shift(2)
    df['vv_t3'] = df['vv'].shift(3)
    df['hv_t3'] = df['hv'].shift(3)

    return df

df = compute_lags(df)

# Filter out nonconsecutive dates
# filtered = []

# for i in t.site.unique():
#     sdf = t[t.site==i]


#     for i in sdf.index:   
#         end = i[1]
#         begin = end - pd.Timedelta(days=24)
#         t = sdf[sdf.index.between(begin, end)]
#         num_points = len(t)
#         if num_points>6:
#             filtered.append(t)


# One hot encode the landcover types 
# one_hot = pd.get_dummies(df.lc_type, drop_first=True )
# X = pd.concat([X, one_hot], axis = 1)


# Modeling options
EPOCHS = int(20e3)
BATCHSIZE = int(2e5)
DROPOUT = 0.1
LOSS = 'mse'

Areg = regularizers.l2(1e-5)
Breg = regularizers.l2(1e-3)
Kreg = regularizers.l2(1e-15)
Rreg = regularizers.l2(1e-15)



# Select dependent variable, drop fluff from input (independent) feats
df = df.dropna()
# df = df.drop([x for x in df.columns if  x in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']])

outdict = {}



for i in df.site.unique():

    print("Processing {}".format(i))

    sdf = df[df.site == i]
    y_test = sdf.rzsm # Psi_RZ? 
    X_test = sdf.drop(["site","rzsm", "ssm", "psi_rz", "psi_s", "precip","lc_type"], axis=1)

    nsdf = df[df.site != i]

    y_train = nsdf.rzsm # Psi_RZ? 
    X_train = nsdf.drop(["site","rzsm", "ssm", "psi_rz", "psi_s", "precip","lc_type"], axis=1)
    
    # Scale inputs
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # reshape input
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
    # y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # create and fit the LSTM network

    input_shape=(X_train.shape[1], X_train.shape[2])

    model = Sequential()

    model.add(LSTM(100, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=0.05,return_sequences=True,bias_regularizer= Breg))
    model.add(LSTM(50, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=0.05,return_sequences=True,bias_regularizer= Breg))
    model.add(LSTM(25, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=0.05,return_sequences=True,bias_regularizer= Breg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=0.05,bias_regularizer= Breg))

    model.add(Dense(1))
    model.compile(loss="mse", optimizer='Nadam')

    # Fit
    model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=1)

    # Get the predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(y_train)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(y_test)

    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    outdict[i] = [y_train, trainPredict, y_test, testPredict]

    plt.scatter(trainY, trainPredict)
    plt.title("training set RMSE = {}".format(round(trainScore,2)))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig("../figures/{}_train_acc.png".format(i))

    plt.scatter(testY, testPredict)
    plt.title("test set RMSE = {}".format(round(testScore,2)))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig("../figures/{}_test_acc.png".format(i))

