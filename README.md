# Time-series-prediction-LSTM
Neural networks such as **long-term and short-term memory (LSTM)** recurrent neural networks can almost perfectly simulate the problem of multiple input variables

This is a great advantage in time series prediction. The classical linear method is difficult to adapt to multivariate or multi input prediction problems.

In this work, you will learn how to develop an LSTM model for multivariate time series prediction in the keras deep learning library.

### In this work,you will know:

1、How to convert the original data set into something we can use for time series prediction.

2、How to prepare data and fit an LSTM model to a multivariable time series prediction problem.

3、How to forecast and readjust the results to the original units

### This work is divided into three parts:

1、Air pollution prediction

2、Basic data preparation

3、Multivariate LSTM prediction model

### Python environment

This tutorial assumes that you have installed the python SciPy environment. This tutorial can use Python 2 or 3. 
You must have keras (version 2.0 or later) installed on the tensorflow or theano backend. 
This tutorial also assumes that you have installed scikit learn, pandas, numpy and Matplotlib libraries.

## 1、Air pollution prediction

The data comes from the hourly weather and air pollution index collected by the U.S. Embassy in Beijing from 2010 to 2014. The dataset includes date, PM2 5 concentration, dew point, temperature, wind direction, wind speed, accumulated hourly snow amount and accumulated hourly rainfall. The complete features in the original data are as follows:

```
1.No :Line number
2.year 
3.month 
4.day 
5.hour 
6.pm2.5 :PM2. 5 concentration
7.DEWP :the dew point
8.TEMP :temperature
9.PRES :Atmospheric pressure
10.cbwd :wind direction
11.lws :wind speed
12.ls :Accumulated snow
13.lr :Cumulative rainfall
```

## 2、Basic data preparation

Before using the data, it is necessary to do some processing on the data. After a rough observation of the data set, it is found that the first 24-hour PM2 5 values are Na, so this part of data needs to be deleted. For a small number of default values at other times, fill them with fillna in pandas; At the same time, it is necessary to integrate date data as an index in pandas.

The following code completes the above process, removes the "no" column from the original data, and names the column with a clearer name.
```
from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution.csv')
```
The processed data is stored in the "aggregation. CSV" file, as follows:
```
                     pollution  dew  temp   press wnd_dir  wnd_spd  snow  rain
date
2010-01-02 00:00:00      129.0  -16  -4.0  1020.0      SE     1.79     0     0
2010-01-02 01:00:00      148.0  -15  -4.0  1020.0      SE     2.68     0     0
2010-01-02 02:00:00      159.0  -11  -5.0  1021.0      SE     3.57     0     0
2010-01-02 03:00:00      181.0   -7  -5.0  1022.0      SE     5.36     1     0
2010-01-02 04:00:00      138.0   -7  -5.0  1022.0      SE     6.25     2     0
```

## 3、Multivariate LSTM prediction model
When using LSTM model, the first step is to adapt the data, including transforming the data set into supervised learning problems and normalized variables (including input and output values), so that it can predict the pollution at the current time (T) through the pollution data of the previous time (t-1) and weather conditions.

The preprocessing module of sklearn is used to encode the category feature "wind direction". Of course, it can also encode the feature one hot. Then all features are normalized, and then the data set is transformed into a supervised learning problem. At the same time, the weather condition features of the current time (T) to be predicted are removed. The complete code is as follows:
```
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
```
By running the above code, you can see the transformed data set, which includes 8 input variables (input characteristics) and 1 output variable (air pollution value and label at the current time t)

Data set processing is relatively simple. There are many ways to try. Some directions to try include:

1. Code the "wind direction" feature;

2. Add seasonal characteristics;

3. The time step exceeds 1.

Among them, the above third method may be the most important for LSTM dealing with time series problems.

Construct LSTM model.
First, we need to divide the processed data set into training set and test set. In order to speed up the training of the model, we only use the data of the first year for training, and then use the remaining 4 years for evaluation.
The following code divides the data set, then divides the training set and test set into input and output variables, and finally transforms the input (x) into the input format of LSTM, namely [samples, timesteps, features].
```
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
```
Run the above code to print the input and output formats of training set and test set:(8760, 1, 8) (8760,) (35039, 1, 8) (35039,)

In the LSTM model, there are 50 neurons in the hidden layer and 1 neuron in the output layer (regression problem). The input variable is a time step (t-1). The loss function adopts mean absolute error (MAE), the optimization algorithm adopts Adam, the model adopts 50 epochs, and the size of each batch is 72.
Finally, set validation in the fit () function_ Data parameter, record the loss of training set and test set, and draw the loss map after completing training and test.
```
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```
Evaluate the effect of the model.
It is worth noting that the predicted results and some test set data need to be combined, and then the scaling needs to be reversed. At the same time, the expected values on the test set also need to be scaled.
After the above processing, the loss is calculated in combination with RMSE (root mean square error).
```
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
```
reference material：https://blog.csdn.net/Together_CZ/article/details/84892130#comments_19305750
