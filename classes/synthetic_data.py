import numpy as np
import matplotlib.pyplot as plt
#import timeseries as ts
#import outlier

from .timeseries import *
from .outlier import *

import os
from pathlib import Path
import shutil
from itertools import product as cartesian_product
import itertools
from collections import namedtuple
import sys
from PIL import Image

PREFIX = "/" + "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[1:-1]) + "/synth_datasets/"
plt.style.use('ggplot')

class SyntheticData():
    """
    Class for generating synthetic datasets with different properties
    """
    def __init__(self, kind, noise=None):
        """
        :param kind: type of signal
        :param noise: standard deviation of gaussian noise
        """
        self.kind = str(kind).lower()
        self.sd_noise = noise

    def generate(self, N_train, N_test, N_outl, width_periods=None):
        """
        generate train, test and outlier data
        :param N_train: number of training samples
        :param N_test: number of test samples
        :param N_outl: number of samples in the outlier dataset
        :param width_periods: width of the period
        :return: (train, test, outl) tuple containing the datasets
        """
        self.N_train = N_train
        self.N_test = N_test
        self.N_outl = N_outl
        self.width = width_periods

        if self.kind == "sinus":
            return self._generateSinus()
        elif self.kind == "square":
            return self._generateSquare()
        elif self.kind == "sawtooth":
            return self._generateSawtooth()
        elif self.kind == "triangle":
            return self._generateTriangle()
        else:
            raise NotImplementedError

    def _generateSinus(self):
        """
        generate a sinus-signal
        :return: (train, test, outl)
        """
        train_signal = HarmonicSignal(frequency=1, noise_std=self.sd_noise, n_samples=self.N_train,
                                         time_end=int(self.N_train / self.width))
        train_ts, train_time = train_signal.generate()
        test_signal = HarmonicSignal(frequency=1, noise_std=self.sd_noise, n_samples=self.N_test,
                                         time_end=int(self.N_test / self.width))
        test_ts, test_time = test_signal.generate()
        outl_signal = HarmonicSignal(frequency=1, noise_std=self.sd_noise, n_samples=self.N_outl,
                                         time_end=int(self.N_outl / self.width))
        outl_ts, outl_time = outl_signal.generate()

        return train_ts, test_ts, outl_ts


    def _generateSquare(self):
        """
        generate square-signal
        :return: (train, test, outl)
        """
        half_period = np.floor(self.width / 2)
        period = [1] * int(half_period) + [-1] * int(half_period)
        train, test, outl = self._repeatPeriodHelper(period)

        return self._addNoiseHelper(train, test, outl)

    def _generateSawtooth(self):
        """
        generate sawtooth-signal
        :return: (train, test, outl)
        """
        period = list(np.linspace(start=-1, stop=1, num=self.width))
        train, test, outl = self._repeatPeriodHelper(period)

        return self._addNoiseHelper(train, test, outl)

    def _generateTriangle(self):
        """
        generate triangle-signal
        :return: (train, test, outl)
        """
        half_period = np.floor(self.width / 2)
        period = list(np.linspace(start=-1, stop=1, num=half_period)) \
                 + list(np.linspace(start=1, stop=-1, num=half_period))
        train, test, outl = self._repeatPeriodHelper(period)

        return self._addNoiseHelper(train, test, outl)

    def _repeatPeriodHelper(self, period):
        """
        helper method for repeating a user-defined period
        :param period: period to be repeated
        :return: (train, test, outl) signals with periodic behaviour as defined in period
        """
        train_signal = period * int(np.ceil(self.N_train / self.width))
        test_signal = period * int(np.ceil(self.N_test / self.width))
        outl_signal = period * int(np.ceil(self.N_outl / self.width))

        return train_signal[:self.N_train], test_signal[:self.N_test], outl_signal[:self.N_outl]

    def _addNoiseHelper(self, train, test, outl):
        """
        helper method for adding noise
        :param train: train signal
        :param test: test signal
        :param outl: outl signal
        :return: noisy (train, test, outl)
        """
        train, test, outl = list(train), list(test), list(outl)
        if self.sd_noise is not None and self.sd_noise != 0:
            train += np.random.normal(0, self.sd_noise, self.N_train)
            test += np.random.normal(0, self.sd_noise, self.N_test)
            outl += np.random.normal(0, self.sd_noise, self.N_outl)

        return train, test, outl

    @classmethod
    def load(cls, dataset=None):
        ds_name = str(dataset)
        if ds_name[-1] != "/":
            ds_name += "/"
        try:
            train = np.load(Path(PREFIX + ds_name + "train.npy"))
            test = np.load(Path(PREFIX + ds_name + "test.npy"))
            outl = np.load(Path(PREFIX + ds_name + "outl.npy"))
            labels = np.load(Path(PREFIX + ds_name + "labels.npy"))
        except IOError:
            print("Wrong dataset name! Exiting ...")
            sys.exit()
        else:
            config = Path(PREFIX + ds_name + "info.txt").read_text()
            img = Image.open(PREFIX + ds_name + "outliers.png")

            print("Read %s with %s" % (dataset, config))
            return (train, test, outl, labels), config, img


if __name__ == "__main__":
    if os.path.exists("../synth_datasets"):
        delete = input("Do you really want to delete all datasets? [Y/N]").lower()
        if delete == "y":
            shutil.rmtree("../synth_datasets")
            os.mkdir("../synth_datasets")
        elif delete == "n":
            print("Not deleting data!")
            #sys.exit()
        else:
            print("Wrong input! Exiting ...")
            sys.exit()
    else:
        os.mkdir("../synth_datasets")

    # define all configurations for dataset generation
    types = ["sinus", "triangle"]
    outl_contextual_size = [20, 35, 50]
    outl_global_size = [15, 5, 2]
    noise = [0.2, 0.5, 1]
    constant = [False]

    contextual_outliers_configs = cartesian_product(["contextual"], types, outl_contextual_size, noise, constant, [1])
    global_outliers_configs = cartesian_product(["global"], types, outl_global_size, noise, constant, [1.5])
    Configuration = namedtuple("Configuration", ["type", "signal", "size", "noise", "constant", "scaling"])
    contextual_configs = [Configuration(*config) for config in contextual_outliers_configs]
    global_configs = [Configuration(*config) for config in global_outliers_configs]

    ds_id = 0
    for _ in range(2):
        for config in global_configs:
        #for config in itertools.chain(contextual_configs, global_configs):
            signal = SyntheticData(kind=config.signal, noise=config.noise)
            train, test, outl = signal.generate(N_train=4500, N_test=500, N_outl=500, width_periods=100)

            outliers = Outlier(type=config.type, n_outliers=1, size=config.size)
            outl_data, outl_idxs = outliers.generate(data=outl, finetuning=[config.scaling], constant=config.constant)
            labels = np.zeros(500)
            plt.plot(outl_data, label="Original")
            for o in outl_idxs:
                plt.plot(o, outl_data[o], label="Outlier")
                labels[o] = True
            plt.legend(loc="upper right")

            # save datasets and outlier-plot to the data-folder
            name_ds = "data" + str(ds_id) +"/"
            os.mkdir(PREFIX + name_ds)
            np.save(Path(PREFIX + name_ds + "train"), train)
            np.save(Path(PREFIX + name_ds + "test"), test)
            np.save(Path(PREFIX + name_ds + "outl"), outl_data)
            np.save(Path(PREFIX + name_ds + "labels"), labels)

            plt.savefig(PREFIX + name_ds + "outliers.png", dpi=200)
            plt.close()

            print("Constructed synthetic dataset %d" % ds_id)

            with open(PREFIX + name_ds + "info.txt", "w") as file:
                file.write(str(config))
            ds_id += 1
