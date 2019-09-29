import timesynth as ts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.fftpack import fft, ifft
import random

def uniform(low, high, n):
  return [random.uniform(low, high) for _ in range(n)]





class Outlier():
  def __init__(self, type, n_outliers, size):
    if type == "global" or type == "contextual":
      self.type = type
    else:
      print("No correct anomaly type")
      exit()

    self.n_outliers = n_outliers
    self.size = size
    self.outliers = []

  def generate(self, data, existing_outliers=[], constant=False, finetuning=None):
    self.existing_outliers = existing_outliers
    if finetuning == None:
      finetuning = np.ones(shape=(self.size, ))
    else:
      finetuning = np.asarray(finetuning)

    if len(data) < (self.n_outliers * self.size):
      print("Too many outliers or too large size")
      exit()

    outlier_data = []
    for o in range(self.n_outliers):
      outlier_start_idx = random.choice([i for i in range(0, len(data) - self.size + 1, self.size)])
      idxs = [i for i in range(outlier_start_idx, outlier_start_idx + self.size)]

      while(set(idxs).intersection(np.array(self.existing_outliers).flatten()) != set()):
        outlier_start_idx = random.choice([i for i in range(0, len(data) - self.size + 1, self.size)])
        idxs = [i for i in range(outlier_start_idx, outlier_start_idx + self.size)]

      #idxs = [idxs[0] - 1] + idxs + [idxs[-1] + 1]
      self.outliers.append(idxs)
      tuning_param = finetuning[o]
      if self.type == "global":
        if np.any(finetuning <= 1.0):
          print("finetuning has to be > 1 for global outliers")
          exit()

        sign = random.choice([+1, -1])
        window = data[outlier_start_idx:outlier_start_idx + self.size]
        if sign == +1:
          to_be_added = max(data) - max(window)
          extremum = np.argmax(window)
        else:
          to_be_added = min(window) - min(data)
          extremum = np.argmin(window)

        if constant:
          generated_outlier = np.ones(shape=(self.size, )) * window[extremum] + to_be_added * sign * tuning_param

        else:
          add = np.array(uniform(low=0, high=to_be_added, n=self.size))
          generated_outlier = np.random.choice(window, self.size, replace=True) + add * sign * tuning_param
          generated_outlier[extremum] = window[extremum] + to_be_added * sign * tuning_param

        outlier_data.append(generated_outlier)

      elif self.type == "contextual":
        if np.any(finetuning > 1.0):
          print("finetuning has to be <= 1 for contextual outliers")
          exit()

        sign = random.choice([+1, -1])
        window = data[outlier_start_idx:outlier_start_idx + self.size]
        if sign == +1:
          to_be_added = max(data) - max(window)
          extremum = np.argmax(window)
        else:
          to_be_added = min(window) - min(data)
          extremum = np.argmin(window)

        if constant:
          generated_outlier = np.ones(self.size) * data[outlier_start_idx]

        else:
          add = np.array(uniform(low=0, high=to_be_added, n=self.size))
          generated_outlier = np.random.choice(window, self.size, replace=True) + add * sign * tuning_param
          generated_outlier[extremum] = window[extremum] + to_be_added * sign * tuning_param

        outlier_data.append(generated_outlier)

    data_with_outliers = np.asarray(data)
    for o in range(0, len(outlier_data)):
      data_with_outliers[self.outliers[o]] = outlier_data[o]


    return data_with_outliers, self.outliers

  def reset(self):
    self.outliers = []



"""global_outliers = Outlier(type="global", n_outliers=5, size=10)
contextual_outliers = Outlier(type="contextual", n_outliers=5, size=10)

#data = np.sin(np.linspace(-np.pi, np.pi, 500))
car = ts.signals.CAR(ar_param=0.9, sigma=0.01)
car_series = ts.TimeSeries(signal_generator=car)

time_sampler = ts.TimeSampler(stop_time=20)
regular_time_samples = time_sampler.sample_regular_time(num_points=500)
data, signals, errors = car_series.sample(regular_time_samples)
print(len(data))
print(data)


global_outliers_data, global_outliers_idxs = global_outliers.generate(data=data)
contextual_outliers_data, contextual_outliers_idxs = contextual_outliers.generate(data=data, existing_outliers=global_outliers_idxs)


data_with_outliers = np.array(data)
for o in range(0, len(global_outliers_data)):
  data_with_outliers[global_outliers_idxs[o]] = global_outliers_data[o]

for o in range(0, len(contextual_outliers_data)):
  data_with_outliers[contextual_outliers_idxs[o]] = contextual_outliers_data[o]

plt.plot([i for i in range(0, len(data_with_outliers))], data_with_outliers, color="red")
plt.plot([i for i in range(0, len(data))], data, color="blue")
plt.show()"""