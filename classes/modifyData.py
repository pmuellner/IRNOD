import numpy as np
import matplotlib.pyplot as plt
import timeseries as ts
import outlier
import pandas as pd
import os
from pathlib import Path

#PREFIX = "/" + "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[1:-1]) + "/modifiedDatasets/ecg1/"
PREFIX = "/" + "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[1:-1]) + "/modifiedDatasets/sunspot/"
os.mkdir(PREFIX)
plt.style.use('ggplot')

if __name__ == "__main__":
    """train_df = pd.read_csv("../datasets/ecg_prep/train1.csv", sep=";", squeeze=True, index_col=0).sort_index()
    test_df = pd.read_csv("../datasets/ecg_prep/test1.csv", sep=";", squeeze=True, index_col=0).sort_index()
    outl_df = pd.read_csv("../datasets/ecg_prep/val1.csv", sep=";", squeeze=True, index_col=0).sort_index()"""
    train_df = pd.read_csv("../datasets/monthly_sunspots/train.csv", sep=";", squeeze=True)
    test_df = pd.read_csv("../datasets/monthly_sunspots/test.csv", sep=";", squeeze=True)
    outl_df = pd.read_csv("../datasets/monthly_sunspots/outl.csv", sep=";", squeeze=True)

    N_outl_ts = 10
    for i in range(N_outl_ts):
        outl = outlier.Outlier(type="contextual", n_outliers=1, size=25)

        outl_data, outl_idxs = outl.generate(data=outl_df.values.copy(), finetuning=[1], constant=False)
        labels = np.zeros(len(outl_df))
        plt.plot(outl_data, label="Original")
        for o in outl_idxs:
            plt.plot(o, outl_data[o], label="Outlier")
            labels[o] = True
        plt.legend(loc="upper right")

        name_ds = "data" + str(i) + "/"
        os.mkdir(PREFIX + name_ds)
        np.save(Path(PREFIX + name_ds + "train"), train_df.values)
        np.save(Path(PREFIX + name_ds + "test"), test_df.values)
        np.save(Path(PREFIX + name_ds + "outl"), outl_data)
        np.save(Path(PREFIX + name_ds + "labels"), labels)

        plt.savefig(PREFIX + name_ds + "outliers.png", dpi=200)
        plt.close()

        print("Constructed synthetic dataset %d" % i)





