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








reference material：https://blog.csdn.net/Together_CZ/article/details/84892130#comments_19305750
