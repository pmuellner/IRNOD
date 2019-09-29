import timesynth as ts
import matplotlib.pyplot as plt

class TimeSampler():
  def __init__(self,n_samples, sample_time_start, sample_time_end):
    self.sample_time_start = sample_time_start
    self.sample_time_end = sample_time_end
    self.n_samples = n_samples

    self.time_sampler = ts.TimeSampler(start_time=self.sample_time_start, stop_time=self.sample_time_end)
    self.time_samples = self.time_sampler.sample_regular_time(num_points=self.n_samples)


class HarmonicSignal(TimeSampler):
  def __init__(self, frequency, n_samples, noise_std=0, time_start=0, time_end=10):
    super(HarmonicSignal, self).__init__(n_samples=n_samples,
                                         sample_time_start=time_start,
                                         sample_time_end=time_end)
    self.frequency = frequency
    self.noise_std = noise_std

  def generate(self):
    sinusoid = ts.signals.Sinusoidal(frequency=self.frequency)
    white_noise = ts.noise.GaussianNoise(std=self.noise_std)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(self.time_samples)

    return samples, self.time_samples

class PseudoPeriodicSignal(TimeSampler):
  def __init__(self, frequency, frequency_std, amplitude_std, n_samples, noise_std=0, time_start=0, time_end=10):
    super(PseudoPeriodicSignal, self).__init__(n_samples=n_samples,
                                               sample_time_start=time_start,
                                               sample_time_end=time_end)
    self.frequency = frequency
    self.noise_std = noise_std
    self.frequency_std = frequency_std
    self.amplitude_std = amplitude_std

  def generate(self):
    pseudo_periodic = ts.signals.PseudoPeriodic(frequency=self.frequency,
                                                freqSD=self.frequency_std,
                                                ampSD=self.amplitude_std)
    white_noise = ts.noise.GaussianNoise(std=self.noise_std)
    timeseries = ts.TimeSeries(pseudo_periodic, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(self.time_samples)

    return samples, self.time_samples


class CarSignal(TimeSampler):
  def __init__(self, ar_param, sigma, n_samples, noise_std=0, time_start=0, time_end=10):
    super(CarSignal, self).__init__(n_samples=n_samples,
                                    sample_time_start=time_start,
                                    sample_time_end=time_end)
    self.ar_param = ar_param
    self.sigma = sigma
    self.noise_std = noise_std

  def generate(self):
    car = ts.signals.CAR(ar_param=self.ar_param, sigma=self.sigma)
    white_noise = ts.noise.GaussianNoise(std=self.noise_std)
    timeseries = ts.TimeSeries(car, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(self.time_samples)

    minimum = min(samples)
    maximum = max(samples)
    scale = lambda elem: (elem - minimum) / (maximum - minimum)

    return scale(samples), self.time_samples


"""harmonic_signal = HarmonicSignal(frequency=0.25, noise_std=0.3, n_samples=500, time_end=20)
harmonic_ts, harmonic_time = harmonic_signal.generate()

pp_signal = PseudoPeriodicSignal(frequency=2, noise_std=0, frequency_std=0.01,
                                 amplitude_std=0.5, n_samples=500,time_end=20)
pp_ts, pp_time = pp_signal.generate()

car_signal = CarSignal(ar_param=0.9, sigma=0.01, noise_std=0, n_samples=500, time_end=20)
car_ts, car_time = car_signal.generate()"""