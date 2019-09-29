import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.io.arff as arff
from pathlib import Path
import sys

def create_time_context(X, context_width=10):
  r_width = np.floor(context_width / 2).astype("int")
  l_width = np.ceil(context_width / 2).astype("int")

  dat = list(X)
  dat_with_ctx = []
  for i in range(l_width, len(dat) - r_width - 1):
    dat_with_ctx.append(dat[i-l_width:i+r_width])

  return np.array(dat_with_ctx)

def create_subset(data, width=5):
  rwidth = int(width / 2)
  lwidth = int(width / 2)

  subsets = []
  for i in range(lwidth, len(data) - rwidth - 1):
    window = data[i-lwidth:i+rwidth+1]
    subsets.append(window)
  return subsets



def create_context(dataset, context_width=10):
  if context_width % 2 != 0:
    print("has to be even")
    exit()
  r_width = int(context_width / 2)
  l_width= int(context_width / 2)
  dataset = dataset.tolist()
  dataX, dataY = [], []
  for i in range(l_width, len(dataset) - r_width - 1):
    d = dataset[i-l_width:i+r_width+1]
    dataX.append(d)
    dataY.append(dataset[i])
  return dataX, dataY

def create_seq_with_context(X, Y, look_back=1):
  dataX, dataY = [], []
  for i in range(len(X)-look_back-1):
    a = X[i:(i+look_back)]
    dataX.append(a)
    dataY.append(X[i+look_back+1])
  return np.array(dataX), np.array(dataY)



def create_seq(dataset, look_back=1):
  dataset = dataset.tolist()
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back)]
    dataX.append(a)
    dataY.append(dataset[i + look_back])
  return np.array(dataX)[..., np.newaxis], np.array(dataY)[..., np.newaxis]

def splitStateOutput(concatenation):
  states = concatenation[:, :, 0::2]
  outputs = concatenation[:, :, 1::2]
  return states, outputs

def calculateSaturation(activations):
  n = len(activations)
  n_right, n_left = 0, 0
  for a in activations:
    if a > 0.9:
      n_right += 1
    if a < 0.1:
      n_left += 1
  return n_left / n, n_right / n

def extractGateStats(x, n_samples, n_hidden, seqlen):
  arguments = []
  for l in range(len(n_hidden)):
    n_neurons = n_hidden[l]
    arguments.append(np.zeros((n_neurons, n_samples, seqlen)))

  for l in range(len(n_hidden)):
    for n in range(n_hidden[l]):
      for t in range(seqlen):
        for s in range(n_samples):
          arguments[l][n, s, t] = x[l][t][s][n]

  return arguments


def flatten(list):
  return [item for sublist in list for item in sublist]

def confidence(train_sat, test_sat):
  train = np.array(flatten(flatten(train_sat)))
  test = np.array(flatten(flatten(test_sat)))

  discrepancy = np.abs(train - test)
  std = np.std(discrepancy)

  n_outliers = 0
  for n in range(len(train)):
    if np.abs(train[n] - test[n]) > 1 * std:
      n_outliers += 1

  return (len(train) - n_outliers) / len(train)

def return_intersection(hist_1, hist_2):
  minima = np.minimum(hist_1, hist_2)
  intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
  return intersection

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def scaler(x, desired_std):
  x = np.asarray(x)
  s = (x - np.mean(x)) / desired_std
  return s

def rolling_sum(a, n=4):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

def getMinMax(*args):
  vmin = np.amin([np.amin(arr) for arr in args])
  vmax = np.amax([np.amax(arr) for arr in args])
  return vmin, vmax

def rollingMean(data, window_size=40, debias=True):
  conv_err = np.convolve(data.flatten(), np.ones(window_size) / window_size, mode="same")
  if debias:
    conv_err -= np.mean(conv_err)
  return conv_err

def averagePrecision(y_true, scores):
  truth = list(np.where(y_true == 1)[0])
  sorted_preds = list(np.argsort(scores.flatten()))[::-1]
  print(sorted_preds)

  precisions = []
  for k in range(len(y_true)):
    top_k_preds = sorted_preds[:k+1]
    p_k = len(set(truth).intersection(set(top_k_preds))) / (k+1)
    precisions.append(p_k)

  return np.sum(precisions) / np.sum(y_true)


