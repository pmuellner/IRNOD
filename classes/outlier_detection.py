from statsmodels.tsa.stattools import acf, pacf, ccf
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from scipy.spatial.distance import cdist
from collections import defaultdict

def extractSeasons(ts):
  data = list(ts)

  autocorr, confint = acf(x=data, alpha=0.5, nlags=len(data)-1)
  signs = np.sign(autocorr)
  sign_change = ((np.roll(signs, 1) - signs) != 0).astype(bool)[1:]

  zeros = np.where(sign_change)[0][::2]
  season_len = np.median([d for d in np.diff(zeros)]).astype(int)
  borders = list(range(zeros[0], len(data), season_len))
  seasons = np.split(np.asarray(data), borders)[1:-1]

  fig, axes = plt.subplots(2,  1)
  axes[0].plot(autocorr, color="orange")
  axes[0].plot(confint, color="green", linestyle="--")
  axes[0].fill_between(range(len(ts)), confint[:, 0], confint[:, 1], color="green", alpha=0.15)
  axes[0].axhline(y=0, color="green", alpha=0.3, linestyle="--")
  axes[1].plot(ts, color="blue")
  for s in borders:
    axes[1].axvline(x=s, color="green")
  plt.show()

  return seasons, borders


def compareSeasons(seasons):
  dist_mat = cdist(seasons, seasons)
  summed_err = np.sum(dist_mat, axis=0)

  return summed_err


def detectOutlierSeason(error_per_season, n_outliers):
  confidence = []
  errs = np.array(error_per_season)
  for o in range(n_outliers):
    identified_outlier_idx = np.argmax(errs)
    clean_idxs = np.arange(0, len(errs)) != identified_outlier_idx
    #score = errs[identified_outlier_idx] / np.median(errs[clean_idxs])
    score = errs[identified_outlier_idx] / np.mean(errs)
    #score = np.abs(errs[identified_outlier_idx] - np.median(errs))
    confidence.append((identified_outlier_idx, score))

    #errs = np.delete(errs, identified_outlier_idx)
    errs[identified_outlier_idx] = -np.inf
  return confidence








