import os
from collections import defaultdict
import pandas as pd

import classes.custom_cells as custom
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import expit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from helpers import create_seq
from classes.synthetic_data import SyntheticData
from itertools import groupby

from sklearn.metrics import classification_report, f1_score, confusion_matrix, average_precision_score
from itertools import product as cartesian_product
from helpers import create_subset

from helpers import averagePrecision
import pprint

from classes.recurrence_plots.plot_recurrence import rec_plot

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.patches as mpatches
plt.style.use('ggplot')

import sys

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

LOF_THRESHOLD = 1.5
ISOF_THRESHOLD = 0.6
RNN_THRESHOLD_FACTOR = 1.5
ISOF_NE = 100
LOF_NN = 20

PLOTS_COMMON_NAME, PLOTS_STATES_NAME, PLOTS_GATES_NAME, PLOTS_RESULTS_NAME = None, None, None, None

def initializeExperiment(type=None):
    prefix = str(type) + "_plots/plots"

    i = 1
    while os.path.exists(prefix + "_" + str(i)):
        i += 1

    global PLOTS_STATES_NAME
    global PLOTS_RESULTS_NAME
    global PLOTS_COMMON_NAME
    global PLOTS_GATES_NAME

    PLOTS_COMMON_NAME = prefix + "_" + str(i) + "/common"
    PLOTS_GATES_NAME = prefix + "_" + str(i) + "/gates"
    PLOTS_STATES_NAME = prefix + "_" + str(i) + "/states"
    PLOTS_RESULTS_NAME = prefix + "_" + str(i) + "/results"

    os.makedirs(PLOTS_COMMON_NAME)
    os.makedirs(PLOTS_GATES_NAME)
    os.makedirs(PLOTS_STATES_NAME)
    os.makedirs(PLOTS_RESULTS_NAME)

    if not os.path.exists("latest_results"):
        os.makedirs("latest_results/")

def getOptimalSubsetLof(data, labels, window_size):
    if window_size == 5:
        subset5 = create_subset(data=data, width=5)
        best_config, f1 = optimizeLOF(X=subset5, labels=labels[3:-2])
        OPT_NN_LOF, OPT_THRESH_LOF = best_config
        clf = LocalOutlierFactor(n_neighbors=OPT_NN_LOF)
        clf.fit(subset5)
        opt_lof5_ts = -clf.negative_outlier_factor_
        return opt_lof5_ts, OPT_THRESH_LOF
    elif window_size == 10:
        subset10 = create_subset(data=data, width=10)
        best_config, f1 = optimizeLOF(X=subset10, labels=labels[6:-5])
        OPT_NN_LOF, OPT_THRESH_LOF = best_config
        clf = LocalOutlierFactor(n_neighbors=OPT_NN_LOF)
        clf.fit(subset10)
        opt_lof10_ts = -clf.negative_outlier_factor_
        return opt_lof10_ts, OPT_THRESH_LOF
    else:
        raise NotImplementedError

def getOptimalSubsetIsof(data, labels, window_size):
    if window_size == 5:
        subset5 = create_subset(data=data, width=5)
        best_config, f1 = optimizeISOF(X=subset5, labels=labels[3:-2])
        OPT_NE_ISOF, OPT_THRESH_ISOF = best_config
        clf = IsolationForest(n_estimators=OPT_NE_ISOF)
        clf.fit(subset5)
        opt_isof5_ts = -clf.score_samples(subset5)
        return opt_isof5_ts, OPT_THRESH_ISOF
    elif window_size == 10:
        subset10 = create_subset(data=data, width=10)
        best_config, f1 = optimizeISOF(X=subset10, labels=labels[6:-5])
        OPT_NE_ISOF, OPT_THRESH_ISOF = best_config
        clf = IsolationForest(n_estimators=OPT_NE_ISOF)
        clf.fit(subset10)
        opt_isof10_ts = -clf.score_samples(subset10)
        return opt_isof10_ts, OPT_THRESH_ISOF
    else:
        raise NotImplementedError


def optimizeLOF(X, labels):
    param_grid = {"n_neighbors": np.arange(5, 40, 5),
                  "threshold": np.arange(1, 5, 0.5)}
    X = np.array(X)

    configs = list(cartesian_product(param_grid["n_neighbors"], param_grid["threshold"]))
    scores = []
    for n_neighbors, threshold in configs:
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm="brute")
        if X.ndim == 1:
            clf.fit(X.reshape(-1, 1))
        else:
            clf.fit(X)
        preds = classify(scores=-clf.negative_outlier_factor_, threshold=threshold)
        f1 = f1_score(y_true=labels, y_pred=preds)
        scores.append(f1)

    best_config_idx = np.argmax(scores)
    best_config = configs[best_config_idx]
    best_f1 = scores[best_config_idx]

    print("=== LOF PARAMS ===")
    print(best_config, best_f1)

    #return (best_config[0], CONST_THRESHOLD), best_f1

    return best_config, best_f1

def optimizeISOF(X, labels):

    """param_grid = {"n_estimators": [5, 10, 20, 25, 35, 50, 80, 100, 150],
                  "threshold": np.arange(1, 5, 0.2)}"""
    param_grid = {"n_estimators": np.arange(5, 120, 10),
                  "threshold": np.arange(0.3, 3, 0.1)}

    configs = list(cartesian_product(param_grid["n_estimators"], param_grid["threshold"]))
    X = np.array(X)
    scores = []
    for n_estimators, threshold in configs:
        clf = IsolationForest(n_estimators=n_estimators)
        if X.ndim == 1:
            clf.fit(X.reshape(-1, 1))
            preds = classify(scores=-clf.score_samples(X.reshape(-1, 1)), threshold=threshold)
        else:
            clf.fit(X)
            preds = classify(scores=-clf.score_samples(X), threshold=threshold)
        f1 = f1_score(y_true=labels, y_pred=preds)
        scores.append(f1)

    best_config_idx = np.argmax(scores)
    best_config = configs[best_config_idx]
    best_f1 = scores[best_config_idx]

    print("=== ISOF PARAMS ===")
    print(best_config)

    return best_config, best_f1

def optimizeRNN(error, labels):
    param_grid = {"threshold": np.arange(1, 5, 0.1)}
    scores = []
    for threshold in param_grid["threshold"]:
        preds = classify(scores=error, threshold=threshold)
        f1 = f1_score(y_pred=preds, y_true=labels)
        scores.append(f1)

    best_config_idx = np.argmax(scores)
    best_config = param_grid["threshold"][best_config_idx]
    best_f1 = scores[best_config_idx]

    print("=== RNN PARAMS ===")
    print(best_config, best_f1)

    return best_config, best_f1



def MinimalRNN(train_ts, test_ts, outl_ts, labels):
    # standardize input data
    scaler = StandardScaler()
    train_ts = scaler.fit_transform(train_ts.reshape(-1, 1))
    test_ts = scaler.transform(test_ts.reshape(-1, 1))
    outl_ts = scaler.transform(outl_ts.reshape(-1, 1))

    train_ts = np.squeeze(train_ts, 1)
    test_ts = np.squeeze(test_ts, 1)
    outl_ts = np.squeeze(outl_ts, 1)

    # construct input features for rnn with a lookback of seqlen
    seqlen = 5
    clean_labels = labels[seqlen+1:]
    X_train, Y_train = create_seq(dataset=train_ts, look_back=seqlen)
    X_test, Y_test = create_seq(dataset=test_ts, look_back=seqlen)
    X_outl, Y_outl = create_seq(dataset=outl_ts, look_back=seqlen)



    X = tf.placeholder(dtype=tf.float64, shape=[None, seqlen, 1])
    Y = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    # define architecture of MinimalRNN
    n_hidden_list = [8]
    stacked_minimalrnn = custom.MinimalRNN(num_units=n_hidden_list)

    rnn_inputs = tf.unstack(X, axis=1)
    all_states, last_state = tf.nn.static_rnn(cell=stacked_minimalrnn, inputs=rnn_inputs, dtype=tf.float64)

    # observe tensors representing update gate
    update_gate_args, W_x = [], []
    for l in range(len(n_hidden_list)):
        update_per_layer = []
        for t in range(seqlen):
            if t == 0:
                name_u = "rnn/update_gate/update_gate/BiasAdd:0"
            else:
                name_u = "rnn/update_gate_" + str(t) + "/update_gate/BiasAdd:0"

            update_per_t = tf.get_default_graph().get_tensor_by_name(name_u)
            update_per_layer.append(update_per_t)

        update_gate_args.append(update_per_layer)
    name_u = "rnn/update_gate/kernel:0"
    U = tf.get_default_graph().get_tensor_by_name(name_u)
    U_h = U[0:n_hidden_list[-1]]
    U_z = U[n_hidden_list[-1]:]

    # build and train network
    W = tf.Variable(dtype=tf.float64, trainable=True, initial_value=np.random.randn(n_hidden_list[-1], 1))
    b = tf.Variable(dtype=tf.float64, initial_value=np.random.randn(1))

    preds = tf.matmul(last_state, W) + b

    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=preds))

    train_step = tf.train.AdamOptimizer(1e-1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    train_loss, test_loss = [], []
    for epoch in range(n_iter):
        if epoch >= 25 and epoch % 25 == 0:
            print(epoch)
        _, train_mse = sess.run([train_step, cost], feed_dict={X: X_train, Y: Y_train})
        train_loss.append(train_mse)

        test_mse = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_loss.append(test_mse)

    plt.plot([i for i in range(len(train_loss))], train_loss, label="Train")
    plt.plot([i for i in range(len(test_loss))], test_loss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/learning_error.png", dpi=200)
    plt.close()

    hypothesis = sess.run(preds, feed_dict={X: X_test, Y: Y_test})
    plt.plot([i for i in range(len(Y_test))], hypothesis, label="Hypothesis", linewidth=2, alpha=1)
    plt.plot([i for i in range(len(Y_test))], Y_test, label="Test", linewidth=1, alpha=0.8)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/hypothesis.png", dpi=200)
    plt.close()

    # calculate and plot the typical rnn forecast prediction error
    hypothesis_outl = sess.run(preds, feed_dict={X: X_outl, Y: Y_outl})
    prediction_error = np.abs(hypothesis_outl - Y_outl)
    normalized_prediction_error = np.abs(prediction_error - np.mean(prediction_error))
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(prediction_error)
    axes[0].set_ylabel("Error")
    axes[0].grid(True)
    axes[1].plot(Y_outl)
    axes[1].set_ylabel("TS")
    plt.savefig(PLOTS_RESULTS_NAME + "/rnn_forecasting_err.png", dpi=200)
    plt.close()

    # optimize and evaluate rnn prediction error
    # TODO research alternative for optimization
    OPT_THRESH_RNN, f1 = optimizeRNN(error=prediction_error, labels=clean_labels)
    preds = classify(scores=prediction_error, threshold=(OPT_THRESH_RNN))
    predictions = {"RNN": preds}
    rnn_ap = average_precision_score(y_true=clean_labels, y_score=prediction_error)
    print("AP RNN: %f" % rnn_ap)
    rnn_ap = averagePrecision(y_true=clean_labels, scores=prediction_error)
    print("myAP RNN: %f" % rnn_ap)

    # evaluate and plot update connections
    U_h = sess.run(U_h, feed_dict={X: X_train, Y: Y_train})
    U_z = sess.run(U_z, feed_dict={X: X_train, Y: Y_train})

    vmin = np.amin(U_h)
    vmax = np.amax(U_h)
    mappable = plt.matshow(U_h, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mappable)
    cbar.set_ticks([vmin, vmin + (vmax - vmin) / 2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/U_h.png", dpi=200)
    plt.close()

    vmin = np.amin(U_z)
    vmax = np.amax(U_z)
    mappable = plt.matshow(U_z, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mappable)
    cbar.set_ticks([vmin, vmin + (vmax - vmin) / 2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/U_z.png", dpi=200)
    plt.close()

    # evaluate and plot state for data including outliers
    last_state_outl = sess.run(last_state, feed_dict={X: X_outl, Y: Y_outl})
    fig, axes = plt.subplots(2, 1)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(np.tanh(last_state_outl[:, n]), label="Neuron %d" % n)

    axes[1].plot(Y_outl)
    plt.tight_layout()
    plt.savefig(PLOTS_STATES_NAME + "/states_outl.png", dpi=200)
    plt.close()

    # compute best lof
    # TODO research alternative for optimization
    best_config, f1 = optimizeLOF(X=Y_outl, labels=clean_labels)
    OPT_NN_LOF, OPT_THRESH_LOF = best_config
    clf = LocalOutlierFactor(n_neighbors=OPT_NN_LOF)
    clf.fit(Y_outl)
    preds = classify(scores=-clf.negative_outlier_factor_, threshold=OPT_THRESH_LOF)
    predictions["LOF_TS"] = preds
    opt_lof_ts = -clf.negative_outlier_factor_
    lofx_ap = average_precision_score(y_true=clean_labels, y_score=opt_lof_ts)
    print("AP LOF X: %f" % lofx_ap)
    lofx_ap = averagePrecision(y_true=clean_labels, scores=opt_lof_ts)
    print("myAP LOF X: %f" % lofx_ap)

    clf = OneClassSVM()
    clf.fit(Y_outl)
    opt_svm_ts = clf.fit_predict(Y_outl)
    predictions["SVM_TS"] = np.logical_and(opt_svm_ts, -1)
    print("SVM AP %f" % (average_precision_score(clean_labels, opt_svm_ts)))

    # compute best isof
    # TODO research alternative for optimization
    best_config, f1 = optimizeISOF(X=Y_outl, labels=clean_labels)
    OPT_NE_ISOF, OPT_THRESH_ISOF = best_config
    clf = IsolationForest(n_estimators=OPT_NE_ISOF)
    clf.fit(Y_outl)
    preds = classify(scores=-clf.score_samples(Y_outl), threshold=OPT_THRESH_ISOF)
    predictions["ISOF_TS"] = preds
    opt_isof_ts = -clf.score_samples(Y_outl)
    print("ISOF AP %f" % (average_precision_score(clean_labels, opt_svm_ts)))


    # compute mean lof and isof over all states
    lof_states_avg_dev, isof_states_avg_dev, svm_states_avg_dev = [], [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].set_ylabel("LOF S N%d" % n)
        axes[2].grid(True)
        lof_states_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].set_ylabel("LOF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = IsolationForest()
        clf.fit(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        isof_score = -clf.score_samples(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        axes[2].plot(isof_score)
        axes[2].set_ylabel("IF S N%d" % n)
        axes[2].grid(True)
        isof_states_avg_dev.append(isof_score)

        axes[3].plot(opt_isof_ts)
        axes[3].set_ylabel("IF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_n" + str(n) + ".png", dpi=200)
        plt.close()

        # svm
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = OneClassSVM()
        clf.fit(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        p = clf.fit_predict(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        axes[2].plot(p)
        axes[2].set_ylabel("SVM S N%d" % n)
        axes[2].grid(True)
        svm_states_avg_dev.append(p)

        axes[3].plot(opt_svm_ts)
        axes[3].set_ylabel("SVM TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/svm_n" + str(n) + ".png", dpi=200)
        plt.close()


    lof_states_avg_dev = -np.mean(lof_states_avg_dev, axis=0)
    isof_states_avg_dev = np.mean(isof_states_avg_dev, axis=0)
    svm_states_avg_dev = np.mean(svm_states_avg_dev, axis=0)
    preds = classify(scores=lof_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_states"] = preds
    preds = classify(scores=isof_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_states"] = preds

    predictions["SVM_states"] = svm_states_avg_dev

    lofh_ap = average_precision_score(y_true=clean_labels, y_score=lof_states_avg_dev)
    print("AP LOF h: %f" % lofh_ap)
    lofh_ap = averagePrecision(y_true=clean_labels, scores=lof_states_avg_dev)
    print("myAP LOF h: %f" % lofh_ap)

    print("AP ISOF h: %f" % (average_precision_score(clean_labels, isof_states_avg_dev)))
    print("AP SVM h: %f" % (average_precision_score(clean_labels, svm_states_avg_dev)))

    # compute mean lof_and isof subset over all states
    lof_5_states_avg_dev, lof_10_states_avg_dev = [], []
    isof_5_states_avg_dev, isof_10_states_avg_dev = [], []
    svm_5_states_avg_dev, svm_10_states_avg_dev = [], []

    opt_lof5_ts, opt_lof5_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_lof10_ts, opt_lof10_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    opt_isof5_ts, opt_isof5_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_isof10_ts, opt_isof10_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    opt_svm5_ts = create_subset(Y_outl.ravel(), width=5)
    opt_svm10_ts = create_subset(Y_outl.ravel(), width=10)

    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[:, n]), width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 S N%d" % n)
        axes[0, 0].grid(True)
        lof_5_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[:, n]), width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 S N%d" % n)
        axes[1, 0].grid(True)
        lof_10_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[:, n]), width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 S N%d" % n)
        axes[0, 0].grid(True)
        isof_5_states_avg_dev.append(-clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[:, n]), width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 S N%d" % n)
        axes[1, 0].grid(True)
        isof_10_states_avg_dev.append(-clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # isof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # svm subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[:, n]), width=5)
        clf = OneClassSVM()
        clf.fit(subset5)
        axes[0, 0].plot(clf.predict(subset5))
        axes[0, 0].set_ylabel("SVM5 S N%d" % n)
        axes[0, 0].grid(True)
        svm_5_states_avg_dev.append(clf.predict(subset5))

        # svm subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[:, n]), width=10)
        clf = OneClassSVM()
        clf.fit(subset10)
        axes[1, 0].plot(clf.predict(subset10))
        axes[1, 0].set_ylabel("SVM10 S N%d" % n)
        axes[1, 0].grid(True)
        svm_10_states_avg_dev.append(clf.predict(subset10))

        # svm subset 5
        axes[0, 1].plot(opt_svm5_ts)
        axes[0, 1].set_ylabel("SVM5 TS N%d" % n)
        axes[0, 1].grid(True)

        # isof subset 10
        axes[1, 1].plot(opt_svm10_ts)
        axes[1, 1].set_ylabel("SVM10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/svm_subset_n" + str(n) + ".png", dpi=200)
        plt.close()



    lof_5_states_avg_dev = -np.mean(lof_5_states_avg_dev, axis=0)
    lof_10_states_avg_dev = -np.mean(lof_10_states_avg_dev, axis=0)
    preds = classify(scores=lof_5_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_states"] = preds
    preds = classify(scores=lof_10_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_states"] = preds

    isof_5_states_avg_dev = np.mean(isof_5_states_avg_dev, axis=0)
    isof_10_states_avg_dev = np.mean(isof_10_states_avg_dev, axis=0)
    preds = classify(scores=isof_5_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_states"] = preds
    preds = classify(scores=isof_10_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_states"] = preds

    svm_5_states_avg_dev = np.mean(svm_5_states_avg_dev, axis=0)
    svm_10_states_avg_dev = np.mean(svm_10_states_avg_dev, axis=0)
    predictions["SVM_5_states"] = svm_5_states_avg_dev
    predictions["SVM_10_states"] = svm_10_states_avg_dev


    # evaluate and plot input argument of update gate
    update_gate_args_outl = sess.run(update_gate_args, feed_dict={X: X_outl, Y: Y_outl})[0]
    update_gate_outl = expit(update_gate_args_outl)

    fig, axes = plt.subplots(2, 1)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(update_gate_outl[-1, :, n], label="Neuron %d" % n)
    axes[1].plot(Y_outl)
    plt.savefig(PLOTS_GATES_NAME + "/gates_outl.png", dpi=200)
    plt.close()

    # compute mean LOF and ISOF over all gates
    lof_gates_avg_dev, isof_gates_avg_dev, svm_gates_avg_dev = [], [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[1].set_ylabel("Gate N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF G N%d" % n)
        lof_gates_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[1].set_ylabel("Gate N%d" % n)

        clf = IsolationForest()
        clf.fit(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[2].plot(-clf.score_samples(update_gate_outl[-1, :, n].reshape(-1, 1)))
        axes[2].grid(True)
        axes[2].set_ylabel("IF G N%d" % n)
        isof_gates_avg_dev.append(-clf.score_samples(update_gate_outl[-1, :, n].reshape(-1, 1)))

        axes[3].plot(opt_isof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[1].set_ylabel("Gate N%d" % n)

        clf = OneClassSVM()
        clf.fit(update_gate_outl[-1, :, n].reshape(-1, 1))
        p = clf.fit_predict(update_gate_outl[-1, :, n].reshape(-1, 1))
        axes[2].plot(p)
        axes[2].grid(True)
        axes[2].set_ylabel("SVM G N%d" % n)
        svm_gates_avg_dev.append(p)

        axes[3].plot(opt_svm_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("SVM TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/svm_n" + str(n) + ".png", dpi=200)
        plt.close()


    lof_gates_avg_dev = -np.mean(lof_gates_avg_dev, axis=0)
    preds = classify(scores=lof_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_gates"] = preds

    isof_gates_avg_dev = np.mean(isof_gates_avg_dev, axis=0)
    preds = classify(scores=isof_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_gates"] = preds

    svm_gates_avg_dev = np.mean(svm_gates_avg_dev, axis=0)
    predictions["SVM_gates"] = svm_gates_avg_dev

    lofu_ap = average_precision_score(y_true=clean_labels, y_score=lof_gates_avg_dev)
    print("AP LOF u: %f" % lofu_ap)
    lofu_ap = averagePrecision(y_true=clean_labels, scores=lof_gates_avg_dev)
    print("myAP LOF u: %f" % lofu_ap)

    print("AP ISOF u: %f" % (average_precision_score(clean_labels, isof_gates_avg_dev)))
    print("AP SVM u: %f" % (average_precision_score(clean_labels, svm_gates_avg_dev)))

    # compute mean lof and isof subset over all gates
    lof_5_gates_avg_dev, lof_10_gates_avg_dev = [], []
    isof_5_gates_avg_dev, isof_10_gates_avg_dev = [], []
    svm_5_gates_avg_dev, svm_10_gates_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=update_gate_outl[-1, :, n], width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 G N%d" % n)
        axes[0, 0].grid(True)
        lof_5_gates_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=update_gate_outl[-1, :, n], width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 G N%d" % n)
        axes[1, 0].grid(True)
        lof_10_gates_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=update_gate_outl[-1, :, n], width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 G N%d" % n)
        axes[0, 0].grid(True)
        isof_5_gates_avg_dev.append(-clf.score_samples(subset5))

        # lof subset 10
        subset10 = create_subset(data=update_gate_outl[-1, :, n], width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 G N%d" % n)
        axes[1, 0].grid(True)
        isof_10_gates_avg_dev.append(-clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # isof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=update_gate_outl[-1, :, n], width=5)
        clf = OneClassSVM()
        clf.fit(subset5)
        axes[0, 0].plot(clf.predict(subset5))
        axes[0, 0].set_ylabel("SVM5 G N%d" % n)
        axes[0, 0].grid(True)
        svm_5_gates_avg_dev.append(clf.predict(subset5))

        # lof subset 10
        subset10 = create_subset(data=update_gate_outl[-1, :, n], width=10)
        clf = OneClassSVM()
        clf.fit(subset10)
        axes[1, 0].plot(clf.predict(subset10))
        axes[1, 0].set_ylabel("SVM10 G N%d" % n)
        axes[1, 0].grid(True)
        svm_10_gates_avg_dev.append(clf.predict(subset10))

        # lof subset 5
        axes[0, 1].plot(opt_svm5_ts)
        axes[0, 1].set_ylabel("SVM5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_svm10_ts)
        axes[1, 1].set_ylabel("SVM10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/svm_subset_n" + str(n) + ".png", dpi=200)
        plt.close()


    lof_5_gates_avg_dev = -np.mean(lof_5_gates_avg_dev, axis=0)
    lof_10_gates_avg_dev = -np.mean(lof_10_gates_avg_dev, axis=0)
    preds = classify(scores=lof_5_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_gates"] = preds
    preds = classify(scores=lof_10_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_gates"] = preds

    isof_5_gates_avg_dev = np.mean(isof_5_gates_avg_dev, axis=0)
    isof_10_gates_avg_dev = np.mean(isof_10_gates_avg_dev, axis=0)
    preds = classify(scores=isof_5_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_gates"] = preds
    preds = classify(scores=isof_10_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_gates"] = preds

    svm_5_gates_avg_dev = np.mean(svm_5_gates_avg_dev, axis=0)
    svm_10_gates_avg_dev = np.mean(svm_10_gates_avg_dev, axis=0)
    predictions["SVM_5_gates"] = svm_5_gates_avg_dev
    predictions["SVM_10_gates"] = svm_10_gates_avg_dev


    # plot a comparison of results for LOF and ISOF
    fig, axes = plt.subplots(4, 1)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(lof_states_avg_dev)
    axes[1].set_ylabel("LOF S")
    axes[1].grid(True)
    axes[2].plot(lof_gates_avg_dev)
    axes[2].set_ylabel("LOF G")
    axes[2].grid(True)
    axes[3].plot(opt_lof_ts)
    axes[3].set_ylabel("LOF TS")
    axes[3].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof__" + arch + ".png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(4, 1)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(isof_states_avg_dev)
    axes[1].set_ylabel("IF S")
    axes[1].grid(True)
    axes[2].plot(isof_gates_avg_dev)
    axes[2].set_ylabel("IF G")
    axes[2].grid(True)
    axes[3].plot(opt_isof_ts)
    axes[3].set_ylabel("IF TS")
    axes[3].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof__" + arch + ".png", dpi=200)
    plt.close()

    # plot a comparison of results for LOF and ISOF subset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(lof_5_states_avg_dev)
    axes[0, 0].set_ylabel("LOF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(lof_10_states_avg_dev)
    axes[0, 1].set_ylabel("LOF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(lof_5_gates_avg_dev)
    axes[1, 0].set_ylabel("LOF5 G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(lof_10_gates_avg_dev)
    axes[1, 1].set_ylabel("LOF10 G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_lof5_ts)
    axes[2, 0].set_ylabel("LOF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_lof5_ts, threshold=opt_lof5_thresh)
    predictions["LOF_5_TS"] = preds

    axes[2, 1].plot(opt_lof10_ts)
    axes[2, 1].set_ylabel("LOF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_lof10_ts, threshold=opt_lof10_thresh)
    predictions["LOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_subset_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof_subset__" + arch + ".png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(isof_5_states_avg_dev)
    axes[0, 0].set_ylabel("IF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(isof_10_states_avg_dev)
    axes[0, 1].set_ylabel("IF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(isof_5_gates_avg_dev)
    axes[1, 0].set_ylabel("IF5 G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(isof_10_gates_avg_dev)
    axes[1, 1].set_ylabel("IF10 G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_isof5_ts)
    axes[2, 0].set_ylabel("IF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_isof5_ts, threshold=opt_isof5_thresh)
    predictions["ISOF_5_TS"] = preds

    axes[2, 1].plot(opt_isof10_ts)
    axes[2, 1].set_ylabel("IF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_isof10_ts, threshold=opt_isof10_thresh)
    predictions["ISOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_subset_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof_subset__" + arch + ".png", dpi=200)
    plt.close()


    # recurrence plots
    for n in range(n_hidden_list[-1]):
        rp = rec_plot(np.tanh(last_state_outl[:, n]))
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_STATES_NAME + "/rec_n" + str(n) + ".png", dpi=200)
        plt.close()

        rp = rec_plot(update_gate_outl[-1, :, n])
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_GATES_NAME + "/rec_n" + str(n) + ".png", dpi=200)
        plt.close()



    # close tensorflow session
    tf.reset_default_graph()
    sess.close()

    return clean_labels, predictions


def classify(scores, threshold):
    classification = scores >= threshold
    return classification

def LSTM(train_ts, test_ts, outl_ts, labels):
    # standardize input data
    scaler = StandardScaler()
    train_ts = scaler.fit_transform(train_ts.reshape(-1, 1))
    test_ts = scaler.transform(test_ts.reshape(-1, 1))
    outl_ts = scaler.transform(outl_ts.reshape(-1, 1))

    train_ts = np.squeeze(train_ts, 1)
    test_ts = np.squeeze(test_ts, 1)
    outl_ts = np.squeeze(outl_ts, 1)

    # construct input features for rnn with lookback of seqlen
    seqlen = 5
    clean_labels = labels[seqlen+1:]
    X_train, Y_train = create_seq(dataset=train_ts, look_back=seqlen)
    X_test, Y_test = create_seq(dataset=test_ts, look_back=seqlen)
    X_outl, Y_outl = create_seq(dataset=outl_ts, look_back=seqlen)

    X = tf.placeholder(dtype=tf.float64, shape=[None, seqlen, 1])
    Y = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    # define architecture of LSTM
    n_hidden_list = [8]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([custom.LSTM(num_units=n, state_is_tuple=False) for n in n_hidden_list])

    # observe tensors representing state, input gate, forget gate, output gate
    rnn_inputs = tf.unstack(X, axis=1)
    state_output_per_t, last_state_output = tf.nn.static_rnn(cell=stacked_lstm, inputs=rnn_inputs, dtype=tf.float64)

    input_gates, forget_gates, output_gates = [], [], []
    for l in range(len(n_hidden_list)):
        input_per_layer, forget_per_layer, output_per_layer = [], [], []
        for t in range(1, 2 * seqlen, 2):
            name_i = "rnn/rnn/multi_rnn_cell/cell_" + str(l) + "/lstm/split_" + str(t) + ":0"
            name_f = "rnn/rnn/multi_rnn_cell/cell_" + str(l) + "/lstm/split_" + str(t) + ":2"
            name_o = "rnn/rnn/multi_rnn_cell/cell_" + str(l) + "/lstm/split_" + str(t) + ":3"

            input_per_t = tf.get_default_graph().get_tensor_by_name(name_i)
            forget_per_t = tf.get_default_graph().get_tensor_by_name(name_f)
            output_per_t = tf.get_default_graph().get_tensor_by_name(name_o)
            input_per_layer.append(input_per_t)
            forget_per_layer.append(forget_per_t)
            output_per_layer.append(output_per_t)

        input_gates.append(input_per_layer)
        forget_gates.append(forget_per_layer)
        output_gates.append(output_per_layer)

    # build and train network
    W = tf.Variable(dtype=tf.float64, trainable=True, initial_value=np.random.randn(n_hidden_list[-1], 1))
    b = tf.Variable(dtype=tf.float64, initial_value=np.random.randn(1))

    preds = tf.matmul(last_state_output[-1][:, 1::2], W) + b

    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=preds))

    train_step = tf.train.AdamOptimizer(1e-1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    train_loss, test_loss = [], []
    for epoch in range(n_iter):
        if epoch >= 25 and epoch % 25 == 0:
            print(epoch)
        _, train_mse = sess.run([train_step, cost], feed_dict={X: X_train, Y: Y_train})
        train_loss.append(train_mse)

        test_mse = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_loss.append(test_mse)

    plt.plot([i for i in range(len(train_loss))], train_loss, label="Train")
    plt.plot([i for i in range(len(test_loss))], test_loss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.savefig(PLOTS_COMMON_NAME + "/learning_error.png", dpi=200)
    plt.close()

    hypothesis = sess.run(preds, feed_dict={X: X_test, Y: Y_test})
    plt.plot([i for i in range(len(Y_test))], hypothesis[0:len(Y_test)], label="Hypothesis", linewidth=2, alpha=1)
    plt.plot([i for i in range(len(Y_test))], Y_test, label="Test", linewidth=1, alpha=0.8)
    plt.legend(loc="upper right")
    plt.savefig(PLOTS_COMMON_NAME + "/hypothesis.png", dpi=200)
    plt.close()

    # calculate and plot the typical rnn forecast prediction error
    hypothesis_outl = sess.run(preds, feed_dict={X: X_outl, Y: Y_outl})
    prediction_error = np.abs(hypothesis_outl - Y_outl)
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(prediction_error)
    axes[0].set_ylabel("Error")
    axes[0].grid(True)
    axes[1].plot(Y_outl)
    axes[1].set_ylabel("TS")
    plt.savefig(PLOTS_RESULTS_NAME + "/rnn_forecasting_err.png", dpi=200)
    plt.close()

    # optimize and evaluate rnn prediction error
    # TODO research alternative for optimization
    OPT_THRESH_RNN, f1 = optimizeRNN(error=prediction_error, labels=clean_labels)
    preds = classify(scores=prediction_error, threshold=OPT_THRESH_RNN)
    predictions = {"RNN": preds}

    # evaluate and plot state for data including outliers
    last_state_output_outl = sess.run(last_state_output, feed_dict={X: X_outl, Y: Y_outl})
    last_state_outl = [l[:, 0::2] for l in last_state_output_outl]

    fig, axes = plt.subplots(len(n_hidden_list) + 1, 1, sharex=True)
    fig.align_ylabels(axes)
    for layer in range(len(n_hidden_list)):
        n_per_layer = n_hidden_list[layer]
        for n in range(n_per_layer):
            axes[layer].plot(np.tanh(last_state_outl[layer][:, n]), label="Neuron %d" % n)
        axes[layer].set_ylabel("Layer %d" % layer)
    axes[-1].plot(Y_outl)
    plt.suptitle("States")
    plt.savefig(PLOTS_STATES_NAME + "/states_outl.png", dpi=200)
    plt.close()

    # compute best lof
    # TODO research alternative for optimization
    best_config, f1 = optimizeLOF(X=Y_outl, labels=clean_labels)
    n_neighbors, OPT_LOF_THRESH = best_config
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    clf.fit(Y_outl)
    opt_lof_ts = -clf.negative_outlier_factor_
    preds = classify(scores=opt_lof_ts, threshold=OPT_LOF_THRESH)
    predictions["LOF_TS"] = preds

    # compute best isof
    # TODO research alternative for optimization
    best_config, f1 = optimizeISOF(X=Y_outl, labels=clean_labels)
    isof_n_estimators, OPT_ISOF_THRESH = best_config
    clf = IsolationForest(n_estimators=isof_n_estimators)
    clf.fit(Y_outl)
    opt_isof_ts = -clf.score_samples(Y_outl)
    preds = classify(scores=opt_isof_ts, threshold=OPT_ISOF_THRESH)
    predictions["ISOF_TS"] = preds

    # compute mean lof and isof over all states
    lof_states_avg_dev, isof_states_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[0][:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(np.tanh(last_state_outl[0][:, n].reshape(-1, 1)))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].set_ylabel("LOF S N%d" % n)
        axes[2].grid(True)
        lof_states_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].set_ylabel("LOF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[0][:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = IsolationForest()
        clf.fit(np.tanh(last_state_outl[0][:, n]).reshape(-1, 1))
        isof_score = -clf.score_samples(np.tanh(last_state_outl[0][:, n]).reshape(-1, 1))
        axes[2].plot(isof_score)
        axes[2].set_ylabel("IF S N%d" % n)
        axes[2].grid(True)
        isof_states_avg_dev.append(isof_score)

        axes[3].plot(opt_isof_ts)
        axes[3].set_ylabel("IF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_states_avg_dev = -np.mean(lof_states_avg_dev, axis=0)
    isof_states_avg_dev = np.mean(isof_states_avg_dev, axis=0)
    preds = classify(scores=lof_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_states"] = preds
    preds = classify(scores=isof_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_states"] = preds

    # compute mean lof_and isof subset over all states
    lof_5_states_avg_dev, lof_10_states_avg_dev = [], []
    isof_5_states_avg_dev, isof_10_states_avg_dev = [], []
    opt_lof5_ts, opt_lof5_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_lof10_ts, opt_lof10_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    opt_isof5_ts, opt_isof5_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_isof10_ts, opt_isof10_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[0][:, n]), width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 S N%d" % n)
        axes[0, 0].grid(True)
        lof_5_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[0][:, n]), width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 S N%d" % n)
        axes[1, 0].grid(True)
        lof_10_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[0][:, n]), width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 S N%d" % n)
        axes[0, 0].grid(True)
        isof_5_states_avg_dev.append(-clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[0][:, n]), width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 S N%d" % n)
        axes[1, 0].grid(True)
        isof_10_states_avg_dev.append(-clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # isof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_5_states_avg_dev = -np.mean(lof_5_states_avg_dev, axis=0)
    lof_10_states_avg_dev = -np.mean(lof_10_states_avg_dev, axis=0)
    preds = classify(scores=lof_5_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_states"] = preds
    preds = classify(scores=lof_10_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_states"] = preds

    isof_5_states_avg_dev = np.mean(isof_5_states_avg_dev, axis=0)
    isof_10_states_avg_dev = np.mean(isof_10_states_avg_dev, axis=0)
    preds = classify(scores=isof_5_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_states"] = preds
    preds = classify(scores=isof_10_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_states"] = preds


    # evaluate and plot gates
    igate_outl, fgate_outl, ogate_outl = sess.run([input_gates, forget_gates, output_gates],
                                                  feed_dict={X: X_outl, Y: Y_outl})

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(expit(igate_outl[0][-1][:, n]), label="Neuron %d" % n)
    axes[1].plot(Y_outl)
    plt.suptitle("Input Gate")
    plt.savefig(PLOTS_GATES_NAME + "/igate_outl.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(expit(fgate_outl[0][-1][:, n]), label="Neuron %d" % n)
    axes[1].plot(Y_outl)
    plt.suptitle("Forget Gate")
    plt.savefig(PLOTS_GATES_NAME + "/fgate_outl.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(expit(ogate_outl[0][-1][:, n]), label="Neuron %d" % n)
    axes[1].plot(Y_outl)
    plt.suptitle("Output Gate")
    plt.savefig(PLOTS_GATES_NAME + "/ogate_outl.png", dpi=200)
    plt.close()

    # compute mean LOF and ISOF over all input gates
    lof_igate_avg_dev, isof_igate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(igate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("igate N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(igate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF G N%d" % n)
        lof_igate_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_igate_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(igate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("igate N%d" % n)

        clf = IsolationForest()
        clf.fit(igate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.score_samples(igate_outl[0][-1][:, n].reshape(-1, 1)))
        axes[2].grid(True)
        axes[2].set_ylabel("IF G N%d" % n)
        isof_igate_avg_dev.append(-clf.score_samples(igate_outl[0][-1][:, n].reshape(-1, 1)))

        axes[3].plot(opt_isof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_igate_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_igate_avg_dev = -np.mean(lof_igate_avg_dev, axis=0)
    preds = classify(scores=lof_igate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_igate"] = preds

    isof_igate_avg_dev = np.mean(isof_igate_avg_dev, axis=0)
    preds = classify(scores=isof_igate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_igate"] = preds

    # compute mean LOF and ISOF over all forget gates
    lof_fgate_avg_dev, isof_fgate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(fgate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("fgate N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(fgate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF G N%d" % n)
        lof_fgate_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_fgate_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(fgate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("fgate N%d" % n)

        clf = IsolationForest()
        clf.fit(fgate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.score_samples(fgate_outl[0][-1][:, n].reshape(-1, 1)))
        axes[2].grid(True)
        axes[2].set_ylabel("IF G N%d" % n)
        isof_fgate_avg_dev.append(-clf.score_samples(fgate_outl[0][-1][:, n].reshape(-1, 1)))

        axes[3].plot(opt_isof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_fgate_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_fgate_avg_dev = -np.mean(lof_fgate_avg_dev, axis=0)
    preds = classify(scores=lof_fgate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_fgate"] = preds

    isof_fgate_avg_dev = np.mean(isof_fgate_avg_dev, axis=0)
    preds = classify(scores=isof_fgate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_fgate"] = preds

    # compute mean LOF and ISOF over all output gates
    lof_ogate_avg_dev, isof_ogate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(ogate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("ogate N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(ogate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF G N%d" % n)
        lof_ogate_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_ogate_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(ogate_outl[0][-1][:, n].reshape(-1, 1))
        axes[1].set_ylabel("ogate N%d" % n)

        clf = IsolationForest()
        clf.fit(ogate_outl[0][-1][:, n].reshape(-1, 1))
        axes[2].plot(-clf.score_samples(ogate_outl[0][-1][:, n].reshape(-1, 1)))
        axes[2].grid(True)
        axes[2].set_ylabel("IF G N%d" % n)
        isof_ogate_avg_dev.append(-clf.score_samples(ogate_outl[0][-1][:, n].reshape(-1, 1)))

        axes[3].plot(opt_isof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_ogate_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_ogate_avg_dev = -np.mean(lof_ogate_avg_dev, axis=0)
    preds = classify(scores=lof_ogate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_ogate"] = preds

    isof_ogate_avg_dev = np.mean(isof_ogate_avg_dev, axis=0)
    preds = classify(scores=isof_ogate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_ogate"] = preds

    # compute mean LOF and ISOF over all gates (multivariable lof, isof)
    lof_all_gates_avg_dev, isof_all_gates_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        multivar = np.vstack([igate_outl[0][-1][:, n], fgate_outl[0][-1][:, n], ogate_outl[0][-1][:, n]]).T
        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")


        clf = LocalOutlierFactor()
        clf.fit(multivar)
        axes[1].plot(-clf.negative_outlier_factor_)
        axes[1].grid(True)
        axes[1].set_ylabel("LOF G N%d" % n)
        lof_all_gates_avg_dev.append(clf.negative_outlier_factor_)

        axes[2].plot(opt_lof_ts)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_all_gates_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        clf = IsolationForest()
        clf.fit(multivar)
        axes[1].plot(-clf.score_samples(multivar))
        axes[1].grid(True)
        axes[1].set_ylabel("IF G N%d" % n)
        isof_all_gates_avg_dev.append(clf.score_samples(multivar))

        axes[2].plot(opt_isof_ts)
        axes[2].grid(True)
        axes[2].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_all_gates_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_all_gates_avg_dev = -np.mean(lof_all_gates_avg_dev, axis=0)
    preds = classify(scores=lof_all_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_all_gates"] = preds

    isof_all_gates_avg_dev = -np.mean(isof_all_gates_avg_dev, axis=0)
    preds = classify(scores=isof_all_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_all_gates"] = preds

    # compute mean lof and isof subset over input gates
    lof_5_igate_avg_dev, lof_10_igate_avg_dev = [], []
    isof_5_igate_avg_dev, isof_10_igate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=igate_outl[0][-1][:, n], width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 G N%d" % n)
        axes[0, 0].grid(True)
        lof_5_igate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=igate_outl[0][-1][:, n], width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 G N%d" % n)
        axes[1, 0].grid(True)
        lof_10_igate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_igate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=igate_outl[0][-1][:, n], width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 G N%d" % n)
        axes[0, 0].grid(True)
        isof_5_igate_avg_dev.append(clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=igate_outl[0][-1][:, n], width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 G N%d" % n)
        axes[1, 0].grid(True)
        isof_10_igate_avg_dev.append(clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_igate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()




    lof_5_igate_avg_dev = -np.mean(lof_5_igate_avg_dev, axis=0)
    lof_10_igate_avg_dev = -np.mean(lof_10_igate_avg_dev, axis=0)
    preds = classify(scores=lof_5_igate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_igate"] = preds
    preds = classify(scores=lof_10_igate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_igate"] = preds

    isof_5_igate_avg_dev = -np.mean(isof_5_igate_avg_dev, axis=0)
    isof_10_igate_avg_dev = -np.mean(isof_10_igate_avg_dev, axis=0)
    preds = classify(scores=isof_5_igate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_igate"] = preds
    preds = classify(scores=isof_10_igate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_igate"] = preds

    # compute mean lof and isof subset over forget gates
    lof_5_fgate_avg_dev, lof_10_fgate_avg_dev = [], []
    isof_5_fgate_avg_dev, isof_10_fgate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=fgate_outl[0][-1][:, n], width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 G N%d" % n)
        axes[0, 0].grid(True)
        lof_5_fgate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=fgate_outl[0][-1][:, n], width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 G N%d" % n)
        axes[1, 0].grid(True)
        lof_10_fgate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_fgate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=fgate_outl[0][-1][:, n], width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 G N%d" % n)
        axes[0, 0].grid(True)
        isof_5_fgate_avg_dev.append(clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=fgate_outl[0][-1][:, n], width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 G N%d" % n)
        axes[1, 0].grid(True)
        isof_10_fgate_avg_dev.append(clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_fgate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()



    lof_5_fgate_avg_dev = -np.mean(lof_5_fgate_avg_dev, axis=0)
    lof_10_fgate_avg_dev = -np.mean(lof_10_fgate_avg_dev, axis=0)
    preds = classify(scores=lof_5_fgate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_fgate"] = preds
    preds = classify(scores=lof_10_fgate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_fgate"] = preds

    isof_5_fgate_avg_dev = -np.mean(isof_5_fgate_avg_dev, axis=0)
    isof_10_fgate_avg_dev = -np.mean(isof_10_fgate_avg_dev, axis=0)
    preds = classify(scores=isof_5_fgate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_fgate"] = preds
    preds = classify(scores=isof_10_fgate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_fgate"] = preds

    # compute mean lof and isof subset over output gates
    lof_5_ogate_avg_dev, lof_10_ogate_avg_dev = [], []
    isof_5_ogate_avg_dev, isof_10_ogate_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=ogate_outl[0][-1][:, n], width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 G N%d" % n)
        axes[0, 0].grid(True)
        lof_5_ogate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=ogate_outl[0][-1][:, n], width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 G N%d" % n)
        axes[1, 0].grid(True)
        lof_10_ogate_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_ogate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=ogate_outl[0][-1][:, n], width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 G N%d" % n)
        axes[0, 0].grid(True)
        isof_5_ogate_avg_dev.append(clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=ogate_outl[0][-1][:, n], width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 G N%d" % n)
        axes[1, 0].grid(True)
        isof_10_ogate_avg_dev.append(clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_ogate_subset_n" + str(n) + ".png", dpi=200)
        plt.close()


    lof_5_ogate_avg_dev = -np.mean(lof_5_ogate_avg_dev, axis=0)
    lof_10_ogate_avg_dev = -np.mean(lof_10_ogate_avg_dev, axis=0)
    preds = classify(scores=lof_5_ogate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_ogate"] = preds
    preds = classify(scores=lof_10_ogate_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_ogate"] = preds

    isof_5_ogate_avg_dev = -np.mean(isof_5_ogate_avg_dev, axis=0)
    isof_10_ogate_avg_dev = -np.mean(isof_10_ogate_avg_dev, axis=0)
    preds = classify(scores=isof_5_ogate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_ogate"] = preds
    preds = classify(scores=isof_10_ogate_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_ogate"] = preds


    # compute mean lof, isof and lof/isof subset over all gates
    lof_mean_iofgate = np.mean(np.vstack([lof_igate_avg_dev, lof_fgate_avg_dev, lof_ogate_avg_dev]), axis=0)
    lof_5_mean_iofgate = np.mean(np.vstack([lof_5_igate_avg_dev, lof_5_fgate_avg_dev, lof_5_ogate_avg_dev]), axis=0)
    lof_10_mean_iofgate = np.mean(np.vstack([lof_10_igate_avg_dev, lof_10_fgate_avg_dev, lof_10_ogate_avg_dev]), axis=0)

    isof_mean_iofgate = np.mean(np.vstack([isof_igate_avg_dev, isof_fgate_avg_dev, isof_ogate_avg_dev]), axis=0)
    isof_5_mean_iofgate = np.mean(np.vstack([isof_5_igate_avg_dev, isof_5_fgate_avg_dev, isof_5_ogate_avg_dev]), axis=0)
    isof_10_mean_iofgate = np.mean(np.vstack([isof_10_igate_avg_dev, isof_10_fgate_avg_dev, isof_10_ogate_avg_dev]), axis=0)



    # plot a comparison of results for LOF
    fig, axes = plt.subplots(5, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(lof_states_avg_dev)
    axes[1].set_ylabel("LOF S")
    axes[1].grid(True)
    axes[2].plot(lof_all_gates_avg_dev)
    axes[2].set_ylabel("LOF all G")
    axes[2].grid(True)
    axes[3].plot(lof_mean_iofgate)
    axes[3].set_ylabel("LOF avg G")

    axes[4].plot(opt_lof_ts)
    axes[4].set_ylabel("LOF TS")
    axes[4].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_lstm__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof__" + arch + ".png", dpi=200)
    plt.close()

    # plot a comparison of results for ISOF
    fig, axes = plt.subplots(5, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(isof_states_avg_dev)
    axes[1].set_ylabel("IF S")
    axes[1].grid(True)
    axes[2].plot(isof_all_gates_avg_dev)
    axes[2].set_ylabel("IF all G")
    axes[2].grid(True)
    axes[3].plot(isof_mean_iofgate)
    axes[3].set_ylabel("IF avg G")
    axes[3].grid(True)

    axes[4].plot(opt_isof_ts)
    axes[4].set_ylabel("IF TS")
    axes[4].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_lstm__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof__" + arch + ".png", dpi=200)
    plt.close()


    fig, axes = plt.subplots(5, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(lof_igate_avg_dev)
    axes[1].set_ylabel("LOF igate")
    axes[1].grid(True)
    axes[2].plot(lof_fgate_avg_dev)
    axes[2].set_ylabel("LOF fgate")
    axes[2].grid(True)
    axes[3].plot(lof_ogate_avg_dev)
    axes[3].set_ylabel("LOF ogate")
    axes[3].grid(True)


    axes[4].plot(opt_lof_ts)
    axes[4].set_ylabel("LOF TS")
    axes[4].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_lstm_gates__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof_gates__" + arch + ".png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(5, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(isof_igate_avg_dev)
    axes[1].set_ylabel("IF igate")
    axes[1].grid(True)
    axes[2].plot(isof_fgate_avg_dev)
    axes[2].set_ylabel("IF fgate")
    axes[2].grid(True)
    axes[3].plot(isof_ogate_avg_dev)
    axes[3].set_ylabel("IF ogate")
    axes[3].grid(True)

    axes[4].plot(opt_isof_ts)
    axes[4].set_ylabel("IF TS")
    axes[4].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_lstm_gates__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof_gates__" + arch + ".png", dpi=200)
    plt.close()



    # plot a comparison of results for LOF subset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(lof_5_states_avg_dev)
    axes[0, 0].set_ylabel("LOF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(lof_10_states_avg_dev)
    axes[0, 1].set_ylabel("LOF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(lof_5_mean_iofgate)
    axes[1, 0].set_ylabel("LOF5 avg G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(lof_10_mean_iofgate)
    axes[1, 1].set_ylabel("LOF10 avg G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_lof5_ts)
    axes[2, 0].set_ylabel("LOF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_lof5_ts, threshold=opt_lof5_thresh)
    predictions["LOF_5_TS"] = preds

    axes[2, 1].plot(opt_lof10_ts)
    axes[2, 1].set_ylabel("LOF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_lof10_ts, threshold=opt_lof10_thresh)
    predictions["LOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_subset_lstm__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof_subset__" + arch + ".png", dpi=200)
    plt.close()

    # plot a comparison of results for ISOF subset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(isof_5_states_avg_dev)
    axes[0, 0].set_ylabel("IF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(isof_10_states_avg_dev)
    axes[0, 1].set_ylabel("IF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(isof_5_mean_iofgate)
    axes[1, 0].set_ylabel("IF5 avg G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(isof_10_mean_iofgate)
    axes[1, 1].set_ylabel("IF10 avg G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_isof5_ts)
    axes[2, 0].set_ylabel("IF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_isof5_ts, threshold=opt_isof5_thresh)
    predictions["ISOF_5_TS"] = preds

    axes[2, 1].plot(opt_isof10_ts)
    axes[2, 1].set_ylabel("IF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_isof10_ts, threshold=opt_isof10_thresh)
    predictions["ISOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_subset_lstm__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof_subset__" + arch + ".png", dpi=200)
    plt.close()

    # recurrence plots
    for n in range(n_hidden_list[-1]):
        rp = rec_plot(np.tanh(last_state_outl[0][:, n]))
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_STATES_NAME + "/rec_n" + str(n) + ".png", dpi=200)
        plt.close()

        rp = rec_plot(igate_outl[0][-1][:, n])
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_GATES_NAME + "/rec_igate_n" + str(n) + ".png", dpi=200)
        plt.close()

        rp = rec_plot(fgate_outl[0][-1][:, n])
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_GATES_NAME + "/rec_fgate_n" + str(n) + ".png", dpi=200)
        plt.close()

        rp = rec_plot(ogate_outl[0][-1][:, n])
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_GATES_NAME + "/rec_ogate_n" + str(n) + ".png", dpi=200)
        plt.close()





    # TODO implement getSaturatingBehaviour in helpers
    """igate_lsat, igate_rsat = getSaturatingBehaviour(gate=igate_outl, n_hidden_list=n_hidden_list, lthresh=0.1,
                                                    rthresh=0.9)
    fgate_lsat, fgate_rsat = getSaturatingBehaviour(gate=igate_outl, n_hidden_list=n_hidden_list, lthresh=0.1,
                                                    rthresh=0.9)
    ogate_lsat, ogate_rsat = getSaturatingBehaviour(gate=igate_outl, n_hidden_list=n_hidden_list, lthresh=0.1,
                                                    rthresh=0.9)

    colors = cm.rainbow(np.linspace(0, 1, len(n_hidden_list)))
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0:2, 0:2])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Input Gate")
    ax2 = plt.subplot(gs[0:2, 2:4])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Forget Gate")
    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("Output Gate")
    patches = []
    for l in range(len(n_hidden_list)):
        for n in range(n_hidden_list[l]):
            ax1.scatter(igate_rsat[l], igate_lsat[l], color=colors[l], alpha=0.7, label="Layer %d" % l)
            ax2.scatter(fgate_rsat[l], fgate_lsat[l], color=colors[l], alpha=0.7, label="Layer %d" % l)
            ax3.scatter(ogate_rsat[l], ogate_lsat[l], color=colors[l], alpha=0.7, label="Layer %d" % l)

        patches.append(mpatches.Patch(color=colors[l], label="Layer %d" % l))

    ax1.legend(handles=patches)
    ax2.legend(handles=patches)
    ax3.legend(handles=patches)
    # plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(PLOTS_GATES_NAME + "/saturations.png", dpi=200)
    plt.show()"""

    # close tensorflow session
    tf.reset_default_graph()
    sess.close()



    return clean_labels, predictions

def MGU(train_ts, test_ts, outl_ts, labels):
    # standardize input data
    scaler = StandardScaler()
    train_ts = scaler.fit_transform(train_ts.reshape(-1, 1))
    test_ts = scaler.transform(test_ts.reshape(-1, 1))
    outl_ts = scaler.transform(outl_ts.reshape(-1, 1))

    train_ts = np.squeeze(train_ts, 1)
    test_ts = np.squeeze(test_ts, 1)
    outl_ts = np.squeeze(outl_ts, 1)

    # construct input features for rnn with a lookback of seqlen
    seqlen = 5
    clean_labels = labels[seqlen + 1:]
    X_train, Y_train = create_seq(dataset=train_ts, look_back=seqlen)
    X_test, Y_test = create_seq(dataset=test_ts, look_back=seqlen)
    X_outl, Y_outl = create_seq(dataset=outl_ts, look_back=seqlen)

    X = tf.placeholder(dtype=tf.float64, shape=[None, seqlen, 1])
    Y = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    # define architecture of Minimal Gated Unit
    n_hidden_list = [8]
    stacked_mgu = custom.MGUCell(num_units=n_hidden_list[-1])

    rnn_inputs = tf.unstack(X, axis=1)
    all_states, last_state = tf.nn.static_rnn(cell=stacked_mgu, inputs=rnn_inputs, dtype=tf.float64)

    # observe tensors representing state and forget gate
    forget_gate_args, W_x = [], []
    for l in range(len(n_hidden_list)):
        forget_per_layer = []
        for t in range(seqlen):
            if t == 0:
                name_f = "rnn/MGUCell/forget_gate/Sigmoid:0"
            else:
                name_f = "rnn/MGUCell_" + str(t) + "/forget_gate/Sigmoid:0"

            forget_per_t = tf.get_default_graph().get_tensor_by_name(name_f)
            forget_per_layer.append(forget_per_t)

        forget_gate_args.append(forget_per_layer)

    # build and train network
    W = tf.Variable(dtype=tf.float64, trainable=True, initial_value=np.random.randn(n_hidden_list[-1], 1))
    b = tf.Variable(dtype=tf.float64, initial_value=np.random.randn(1))

    preds = tf.matmul(last_state, W) + b

    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=preds))

    train_step = tf.train.AdamOptimizer(1e-1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    train_loss, test_loss = [], []
    for epoch in range(n_iter):
        if epoch >= 25 and epoch % 25 == 0:
            print(epoch)
        _, train_mse = sess.run([train_step, cost], feed_dict={X: X_train, Y: Y_train})
        train_loss.append(train_mse)

        test_mse = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_loss.append(test_mse)

    plt.plot([i for i in range(len(train_loss))], train_loss, label="Train")
    plt.plot([i for i in range(len(test_loss))], test_loss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/learning_error.png", dpi=200)
    plt.close()

    hypothesis = sess.run(preds, feed_dict={X: X_test, Y: Y_test})
    plt.plot([i for i in range(len(Y_test))], hypothesis[0:len(Y_test)], label="Hypothesis", linewidth=2, alpha=1)
    plt.plot([i for i in range(len(Y_test))], Y_test, label="Test", linewidth=1, alpha=0.8)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_COMMON_NAME + "/hypothesis.png", dpi=200)
    plt.close()

    # calculate and plot the typical rnn forecast prediction error
    hypothesis_outl = sess.run(preds, feed_dict={X: X_outl, Y: Y_outl})
    prediction_error = np.abs(hypothesis_outl - Y_outl)
    normalized_prediction_error = np.abs(prediction_error - np.mean(prediction_error))
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.align_ylabels(axes)
    axes[0].plot(prediction_error)
    axes[0].set_ylabel("Error")
    axes[0].grid(True)
    axes[1].plot(Y_outl)
    axes[1].set_ylabel("TS")
    plt.savefig(PLOTS_RESULTS_NAME + "/rnn_forecasting_err.png", dpi=200)
    plt.close()

    # optimize and evaluate rnn prediction error
    # TODO research alternative for optimization
    threshold, f1 = optimizeRNN(error=prediction_error, labels=clean_labels)
    preds = classify(scores=prediction_error, threshold=threshold)
    predictions = {"RNN": preds}

    # evaluate and plot state for data including outliers
    last_state_outl = sess.run(last_state, feed_dict={X: X_outl, Y: Y_outl})
    fig, axes = plt.subplots(2, 1)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(np.tanh(last_state_outl[:, n]), label="Neuron %d" % n)

    axes[1].plot(Y_outl)
    plt.tight_layout()
    plt.savefig(PLOTS_STATES_NAME + "/states_outl.png", dpi=200)
    plt.close()

    # compute best lof
    # TODO research alternative for optimization
    best_config, f1 = optimizeLOF(X=Y_outl, labels=clean_labels)
    n_neighbors, threshold = best_config
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    clf.fit(Y_outl)
    opt_lof_ts = -clf.negative_outlier_factor_
    preds = classify(scores=opt_lof_ts, threshold=threshold)
    predictions["LOF_TS"] = preds

    # compute best isof
    # TODO research alternative for optimization
    best_config, f1 = optimizeISOF(X=Y_outl, labels=clean_labels)
    isof_n_estimators, isof_threshold = best_config
    clf = IsolationForest(n_estimators=isof_n_estimators)
    clf.fit(Y_outl)
    opt_isof_ts = -clf.score_samples(Y_outl)
    preds = classify(scores=opt_isof_ts, threshold=isof_threshold)
    predictions["ISOF_TS"] = preds

    # compute mean lof and isof over all states
    lof_states_avg_dev, isof_states_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].set_ylabel("LOF S N%d" % n)
        axes[2].grid(True)
        lof_states_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].set_ylabel("LOF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)
        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(np.tanh(last_state_outl[:, n]))
        axes[1].set_ylabel("State N%d" % n)

        clf = IsolationForest()
        clf.fit(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        isof_score = -clf.score_samples(np.tanh(last_state_outl[:, n].reshape(-1, 1)))
        axes[2].plot(isof_score)
        axes[2].set_ylabel("IF S N%d" % n)
        axes[2].grid(True)
        isof_states_avg_dev.append(isof_score)

        axes[3].plot(opt_isof_ts)
        axes[3].set_ylabel("IF TS N%d" % n)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_states_avg_dev = -np.mean(lof_states_avg_dev, axis=0)
    isof_states_avg_dev = np.mean(isof_states_avg_dev, axis=0)
    preds = classify(scores=lof_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_states"] = preds
    preds = classify(scores=isof_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_states"] = preds

    # compute mean lof_and isof subset over all states
    lof_5_states_avg_dev, lof_10_states_avg_dev = [], []
    isof_5_states_avg_dev, isof_10_states_avg_dev = [], []
    opt_lof5_ts, opt_lof5_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_lof10_ts, opt_lof10_thresh = getOptimalSubsetLof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    opt_isof5_ts, opt_isof5_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=5)
    opt_isof10_ts, opt_isof10_thresh = getOptimalSubsetIsof(data=Y_outl.ravel(), labels=clean_labels, window_size=10)
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[:, n]), width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 S N%d" % n)
        axes[0, 0].grid(True)
        lof_5_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[:, n]), width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 S N%d" % n)
        axes[1, 0].grid(True)
        lof_10_states_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/lof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)
        # isof subset 5
        subset5 = create_subset(data=np.tanh(last_state_outl[:, n]), width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 S N%d" % n)
        axes[0, 0].grid(True)
        isof_5_states_avg_dev.append(-clf.score_samples(subset5))

        # isof subset 10
        subset10 = create_subset(data=np.tanh(last_state_outl[:, n]), width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 S N%d" % n)
        axes[1, 0].grid(True)
        isof_10_states_avg_dev.append(-clf.score_samples(subset10))

        # isof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # isof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_STATES_NAME + "/isof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_5_states_avg_dev = -np.mean(lof_5_states_avg_dev, axis=0)
    lof_10_states_avg_dev = -np.mean(lof_10_states_avg_dev, axis=0)
    preds = classify(scores=lof_5_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_states"] = preds
    preds = classify(scores=lof_10_states_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_states"] = preds

    isof_5_states_avg_dev = np.mean(isof_5_states_avg_dev, axis=0)
    isof_10_states_avg_dev = np.mean(isof_10_states_avg_dev, axis=0)
    preds = classify(scores=isof_5_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_states"] = preds
    preds = classify(scores=isof_10_states_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_states"] = preds

    # evaluate and plot input argument of update gate
    forget_gate_args_outl = sess.run(forget_gate_args, feed_dict={X: X_outl, Y: Y_outl})[0]
    forget_gate_outl = expit(forget_gate_args_outl)

    fig, axes = plt.subplots(2, 1)
    fig.align_ylabels(axes)
    for n in range(n_hidden_list[-1]):
        axes[0].plot(forget_gate_outl[-1, :, n], label="Neuron %d" % n)
    axes[1].plot(Y_outl)
    plt.savefig(PLOTS_GATES_NAME + "/gates_outl.png", dpi=200)
    plt.close()

    # compute mean LOF and ISOF over all gates
    lof_gates_avg_dev, isof_gates_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(forget_gate_outl[-1, :, n].reshape(-1, 1))
        axes[1].set_ylabel("Gate N%d" % n)

        clf = LocalOutlierFactor()
        clf.fit(forget_gate_outl[-1, :, n].reshape(-1, 1))
        axes[2].plot(-clf.negative_outlier_factor_)
        axes[2].grid(True)
        axes[2].set_ylabel("LOF G N%d" % n)
        lof_gates_avg_dev.append(clf.negative_outlier_factor_)

        axes[3].plot(opt_lof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("LOF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(4, 1, sharex=True)
        fig.align_ylabels(axes)

        axes[0].plot(Y_outl)
        axes[0].set_ylabel("TS")

        axes[1].plot(forget_gate_outl[-1, :, n].reshape(-1, 1))
        axes[1].set_ylabel("Gate N%d" % n)

        clf = IsolationForest()
        clf.fit(forget_gate_outl[-1, :, n].reshape(-1, 1))
        axes[2].plot(-clf.score_samples(forget_gate_outl[-1, :, n].reshape(-1, 1)))
        axes[2].grid(True)
        axes[2].set_ylabel("IF G N%d" % n)
        isof_gates_avg_dev.append(-clf.score_samples(forget_gate_outl[-1, :, n].reshape(-1, 1)))

        axes[3].plot(opt_isof_ts)
        axes[3].grid(True)
        axes[3].set_ylabel("IF TS")

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_gates_avg_dev = -np.mean(lof_gates_avg_dev, axis=0)
    preds = classify(scores=lof_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_gates"] = preds

    isof_gates_avg_dev = np.mean(isof_gates_avg_dev, axis=0)
    preds = classify(scores=isof_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_gates"] = preds

    # compute mean lof and isof subset over all gates
    lof_5_gates_avg_dev, lof_10_gates_avg_dev = [], []
    isof_5_gates_avg_dev, isof_10_gates_avg_dev = [], []
    for n in range(n_hidden_list[-1]):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=forget_gate_outl[-1, :, n], width=5)
        clf = LocalOutlierFactor()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.negative_outlier_factor_)
        axes[0, 0].set_ylabel("LOF5 G N%d" % n)
        axes[0, 0].grid(True)
        lof_5_gates_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 10
        subset10 = create_subset(data=forget_gate_outl[-1, :, n], width=10)
        clf = LocalOutlierFactor()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.negative_outlier_factor_)
        axes[1, 0].set_ylabel("LOF10 G N%d" % n)
        axes[1, 0].grid(True)
        lof_10_gates_avg_dev.append(clf.negative_outlier_factor_)

        # lof subset 5
        axes[0, 1].plot(opt_lof5_ts)
        axes[0, 1].set_ylabel("LOF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_lof10_ts)
        axes[1, 1].set_ylabel("LOF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/lof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.align_ylabels(axes)

        # lof subset 5
        subset5 = create_subset(data=forget_gate_outl[-1, :, n], width=5)
        clf = IsolationForest()
        clf.fit(subset5)
        axes[0, 0].plot(-clf.score_samples(subset5))
        axes[0, 0].set_ylabel("IF5 G N%d" % n)
        axes[0, 0].grid(True)
        isof_5_gates_avg_dev.append(-clf.score_samples(subset5))

        # lof subset 10
        subset10 = create_subset(data=forget_gate_outl[-1, :, n], width=10)
        clf = IsolationForest()
        clf.fit(subset10)
        axes[1, 0].plot(-clf.score_samples(subset10))
        axes[1, 0].set_ylabel("IF10 G N%d" % n)
        axes[1, 0].grid(True)
        isof_10_gates_avg_dev.append(-clf.score_samples(subset10))

        # lof subset 5
        axes[0, 1].plot(opt_isof5_ts)
        axes[0, 1].set_ylabel("IF5 TS N%d" % n)
        axes[0, 1].grid(True)

        # lof subset 10
        axes[1, 1].plot(opt_isof10_ts)
        axes[1, 1].set_ylabel("IF10 TS N%d" % n)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_GATES_NAME + "/isof_subset_n" + str(n) + ".png", dpi=200)
        plt.close()

    lof_5_gates_avg_dev = -np.mean(lof_5_gates_avg_dev, axis=0)
    lof_10_gates_avg_dev = -np.mean(lof_10_gates_avg_dev, axis=0)
    preds = classify(scores=lof_5_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_5_gates"] = preds
    preds = classify(scores=lof_10_gates_avg_dev, threshold=LOF_THRESHOLD)
    predictions["LOF_10_gates"] = preds

    isof_5_gates_avg_dev = np.mean(isof_5_gates_avg_dev, axis=0)
    isof_10_gates_avg_dev = np.mean(isof_10_gates_avg_dev, axis=0)
    preds = classify(scores=isof_5_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_5_gates"] = preds
    preds = classify(scores=isof_10_gates_avg_dev, threshold=ISOF_THRESHOLD)
    predictions["ISOF_10_gates"] = preds

    # plot a comparison of results for LOF and ISOF
    fig, axes = plt.subplots(4, 1)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(lof_states_avg_dev)
    axes[1].set_ylabel("LOF S")
    axes[1].grid(True)
    axes[2].plot(lof_gates_avg_dev)
    axes[2].set_ylabel("LOF G")
    axes[2].grid(True)

    axes[3].plot(opt_lof_ts)
    axes[3].set_ylabel("LOF TS")
    axes[3].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof__" + arch + ".png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(4, 1)
    fig.align_ylabels(axes)
    axes[0].plot(Y_outl)
    axes[0].set_ylabel("TS")
    axes[1].plot(isof_states_avg_dev)
    axes[1].set_ylabel("IF S")
    axes[1].grid(True)
    axes[2].plot(isof_gates_avg_dev)
    axes[2].set_ylabel("IF G")
    axes[2].grid(True)

    axes[3].plot(opt_isof_ts)
    axes[3].set_ylabel("IF TS")
    axes[3].grid(True)

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof__" + arch + ".png", dpi=200)
    plt.close()

    # plot a comparison of results for LOF and ISOF subset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(lof_5_states_avg_dev)
    axes[0, 0].set_ylabel("LOF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(lof_10_states_avg_dev)
    axes[0, 1].set_ylabel("LOF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(lof_5_gates_avg_dev)
    axes[1, 0].set_ylabel("LOF5 G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(lof_10_gates_avg_dev)
    axes[1, 1].set_ylabel("LOF10 G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_lof5_ts)
    axes[2, 0].set_ylabel("LOF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_lof5_ts, threshold=opt_lof5_thresh)
    predictions["LOF_5_TS"] = preds

    axes[2, 1].plot(opt_lof10_ts)
    axes[2, 1].set_ylabel("LOF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_lof10_ts, threshold=opt_lof10_thresh)
    predictions["LOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/lof_subset_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/lof_subset__" + arch + ".png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    fig.align_ylabels(axes)
    axes[0, 0].plot(isof_5_states_avg_dev)
    axes[0, 0].set_ylabel("IF5 S")
    axes[0, 0].grid(True)
    axes[0, 1].plot(isof_10_states_avg_dev)
    axes[0, 1].set_ylabel("IF10 S")
    axes[0, 1].grid(True)
    axes[1, 0].plot(isof_5_gates_avg_dev)
    axes[1, 0].set_ylabel("IF5 G")
    axes[1, 0].grid(True)
    axes[1, 1].plot(isof_10_gates_avg_dev)
    axes[1, 1].set_ylabel("IF10 G")
    axes[1, 1].grid(True)

    axes[2, 0].plot(opt_isof5_ts)
    axes[2, 0].set_ylabel("IF5 TS")
    axes[2, 0].grid(True)
    preds = classify(scores=opt_isof5_ts, threshold=opt_isof5_thresh)
    predictions["ISOF_5_TS"] = preds

    axes[2, 1].plot(opt_isof10_ts)
    axes[2, 1].set_ylabel("IF10 TS")
    axes[2, 1].grid(True)
    preds = classify(scores=opt_isof10_ts, threshold=opt_isof10_thresh)
    predictions["ISOF_10_TS"] = preds

    arch = "_".join(map(str, n_hidden_list))
    plt.tight_layout()
    plt.savefig("latest_results/isof_subset_minimalRNN__" + arch + ".png", dpi=200)
    plt.savefig(PLOTS_RESULTS_NAME + "/isof_subset__" + arch + ".png", dpi=200)
    plt.close()

    # recurrence plots
    for n in range(n_hidden_list[-1]):
        rp = rec_plot(np.tanh(last_state_outl[:, n]))
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_STATES_NAME + "/rec_n" + str(n) + ".png", dpi=200)
        plt.close()

        rp = rec_plot(forget_gate_outl[-1, :, n])
        plt.imshow(rp, cmap='gray')
        plt.savefig(PLOTS_GATES_NAME + "/rec_n" + str(n) + ".png", dpi=200)
        plt.close()





    # close tensorflow session
    tf.reset_default_graph()
    sess.close()

    return clean_labels, predictions

def printEval(reports):
    for key in reports:
        print("\n=== REPORT " + key.upper() + " ===")
        #print(reports[key])
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(reports[key])
        print()




def runExperiment(train_ts, test_ts, outl_ts, labels, type=None):
    rp = rec_plot(outl_ts)
    plt.imshow(rp, cmap='gray')
    plt.savefig(PLOTS_COMMON_NAME + "/rec_ts.png", dpi=200)
    plt.close()



    if type == "MinimalRNN":
        clean_labels, predictions = MinimalRNN(train_ts, test_ts, outl_ts, labels)
        report_rnn = evaluate(preds=predictions["RNN"], labels=clean_labels)

        report_lof = evaluate(preds=predictions["LOF_TS"], labels=clean_labels)
        report_lof_states = evaluate(preds=predictions["LOF_states"], labels=clean_labels)
        report_lof_gates = evaluate(preds=predictions["LOF_gates"], labels=clean_labels)
        report_lof5_ts = evaluate(preds=predictions["LOF_5_TS"], labels=clean_labels[3:-2])
        report_lof10_ts = evaluate(preds=predictions["LOF_10_TS"], labels=clean_labels[6:-5])
        report_lof5_states = evaluate(preds=predictions["LOF_5_states"], labels=clean_labels[3:-2])
        report_lof10_states = evaluate(preds=predictions["LOF_10_states"], labels=clean_labels[6:-5])
        report_lof5_gates = evaluate(preds=predictions["LOF_5_gates"], labels=clean_labels[3:-2])
        report_lof10_gates = evaluate(preds=predictions["LOF_10_gates"], labels=clean_labels[6:-5])

        report_isof_ts = evaluate(preds=predictions["ISOF_TS"], labels=clean_labels)
        report_isof_states = evaluate(preds=predictions["ISOF_states"], labels=clean_labels)
        report_isof_gates = evaluate(preds=predictions["ISOF_gates"], labels=clean_labels)
        report_isof5_ts = evaluate(preds=predictions["ISOF_5_TS"], labels=clean_labels[3:-2])
        report_isof10_ts = evaluate(preds=predictions["ISOF_10_TS"], labels=clean_labels[6:-5])
        report_isof5_states = evaluate(preds=predictions["ISOF_5_states"], labels=clean_labels[3:-2])
        report_isof10_states = evaluate(preds=predictions["ISOF_10_states"], labels=clean_labels[6:-5])
        report_isof5_gates = evaluate(preds=predictions["ISOF_5_gates"], labels=clean_labels[3:-2])
        report_isof10_gates = evaluate(preds=predictions["ISOF_10_gates"], labels=clean_labels[6:-5])

        report_svm_ts = evaluate(preds=predictions["SVM_TS"], labels=clean_labels)
        report_svm_states = evaluate(preds=predictions["SVM_states"], labels=clean_labels)





        all_reports = {"rnn": report_rnn,
                   "lof_ts": report_lof,
                   "lof_states": report_lof_states,
                   "lof_gates": report_lof_gates,
                   "lof_5_states": report_lof5_states,
                   "lof_10_states": report_lof10_states,
                   "lof_5_gates": report_lof5_gates,
                   "lof_10_gates": report_lof10_gates,
                   "lof_5_ts": report_lof5_ts,
                   "lof_10_ts": report_lof10_ts,
                   "isof_states": report_isof_states,
                   "isof_gates": report_isof_gates,
                   "isof_5_states": report_isof5_states,
                   "isof_10_states": report_isof10_states,
                   "isof_5_gates": report_isof5_gates,
                   "isof_10_gates": report_isof10_gates,
                   "isof_ts": report_isof_ts,
                   "isof5_ts": report_isof5_ts,
                   "isof10_ts": report_isof10_ts}

        printEval(all_reports)
        return all_reports

    elif type == "LSTM":
        clean_labels, predictions = LSTM(train_ts, test_ts, outl_ts, labels)
        report_rnn = evaluate(preds=predictions["RNN"], labels=clean_labels)
        report_lof_ts = evaluate(preds=predictions["LOF_TS"], labels=clean_labels)
        report_lof_states = evaluate(preds=predictions["LOF_states"], labels=clean_labels)
        report_lof5_states = evaluate(preds=predictions["LOF_5_states"], labels=clean_labels[3:-2])
        report_lof10_states = evaluate(preds=predictions["LOF_10_states"], labels=clean_labels[6:-5])
        report_lof_igate = evaluate(preds=predictions["LOF_igate"], labels=clean_labels)
        report_lof_fgate = evaluate(preds=predictions["LOF_fgate"], labels=clean_labels)
        report_lof_ogate = evaluate(preds=predictions["LOF_ogate"], labels=clean_labels)
        report_lof_all_gates = evaluate(preds=predictions["LOF_all_gates"], labels=clean_labels)
        report_lof5_igate = evaluate(preds=predictions["LOF_5_igate"], labels=clean_labels[3:-2])
        report_lof10_igate = evaluate(preds=predictions["LOF_10_igate"], labels=clean_labels[6:-5])
        report_lof5_fgate = evaluate(preds=predictions["LOF_5_fgate"], labels=clean_labels[3:-2])
        report_lof10_fgate = evaluate(preds=predictions["LOF_10_fgate"], labels=clean_labels[6:-5])
        report_lof5_ogate = evaluate(preds=predictions["LOF_5_ogate"], labels=clean_labels[3:-2])
        report_lof10_ogate = evaluate(preds=predictions["LOF_10_ogate"], labels=clean_labels[6:-5])
        report_lof5_ts = evaluate(preds=predictions["LOF_5_TS"], labels=clean_labels[3:-2])
        report_lof10_ts = evaluate(preds=predictions["LOF_10_TS"], labels=clean_labels[6:-5])

        report_isof_ts = evaluate(preds=predictions["ISOF_TS"], labels=clean_labels)
        report_isof_states = evaluate(preds=predictions["ISOF_states"], labels=clean_labels)
        report_isof5_states = evaluate(preds=predictions["ISOF_5_states"], labels=clean_labels[3:-2])
        report_isof10_states = evaluate(preds=predictions["ISOF_10_states"], labels=clean_labels[6:-5])
        report_isof_igate = evaluate(preds=predictions["ISOF_igate"], labels=clean_labels)
        report_isof_fgate = evaluate(preds=predictions["ISOF_fgate"], labels=clean_labels)
        report_isof_ogate = evaluate(preds=predictions["ISOF_ogate"], labels=clean_labels)
        report_isof_all_gates = evaluate(preds=predictions["ISOF_all_gates"], labels=clean_labels)
        report_isof5_igate = evaluate(preds=predictions["ISOF_5_igate"], labels=clean_labels[3:-2])
        report_isof10_igate = evaluate(preds=predictions["ISOF_10_igate"], labels=clean_labels[6:-5])
        report_isof5_fgate = evaluate(preds=predictions["ISOF_5_fgate"], labels=clean_labels[3:-2])
        report_isof10_fgate = evaluate(preds=predictions["ISOF_10_fgate"], labels=clean_labels[6:-5])
        report_isof5_ogate = evaluate(preds=predictions["ISOF_5_ogate"], labels=clean_labels[3:-2])
        report_isof10_ogate = evaluate(preds=predictions["ISOF_10_ogate"], labels=clean_labels[6:-5])
        report_isof5_ts = evaluate(preds=predictions["ISOF_5_TS"], labels=clean_labels[3:-2])
        report_isof10_ts = evaluate(preds=predictions["ISOF_10_TS"], labels=clean_labels[6:-5])
        

        all_reports = {"report_rnn": report_rnn,
                       "report_lof_ts": report_lof_ts,
                       "report_lof_states": report_lof_states,
                       "report_lof5_states": report_lof5_states,
                       "report_lof10_states": report_lof10_states,
                       "report_lof_igate": report_lof_igate,
                       "report_lof_fgate": report_lof_fgate,
                       "report_lof_ogate": report_lof_ogate,
                       "report_lof_all_gates": report_lof_all_gates,
                       "report_lof5_igate": report_lof5_igate,
                       "report_lof10_igate": report_lof10_igate,
                       "report_lof5_fgate": report_lof5_fgate,
                       "report_lof10_fgate": report_lof10_fgate,
                       "report_lof5_ogate": report_lof5_ogate,
                       "report_lof10_ogate": report_lof10_ogate,
                       "report_lof5_ts": report_lof5_ts,
                       "report_lof10_ts": report_lof10_ts,
                       "report_isof_ts": report_isof_ts,
                       "report_isof_states": report_isof_states,
                       "report_isof5_states": report_isof5_states,
                       "report_isof10_states": report_isof10_states,
                       "report_isof_igate": report_isof_igate,
                       "report_isof_fgate": report_isof_fgate,
                       "report_isof_ogate": report_isof_ogate,
                       "report_isof_all_gates": report_isof_all_gates,
                       "report_isof5_igate": report_isof5_igate,
                       "report_isof10_igate": report_isof10_igate,
                       "report_isof5_fgate": report_isof5_fgate,
                       "report_isof10_fgate": report_isof10_fgate,
                       "report_isof5_ogate": report_isof5_ogate,
                       "report_isof10_ogate": report_isof10_ogate,
                       "report_isof5_ts": report_isof5_ts,
                       "report_isof10_ts": report_isof10_ts}
        printEval(all_reports)
        return all_reports

    elif type == "MGU":
        clean_labels, predictions = MGU(train_ts, test_ts, outl_ts, labels)
        report_rnn = evaluate(preds=predictions["RNN"], labels=clean_labels)

        report_lof = evaluate(preds=predictions["LOF_TS"], labels=clean_labels)
        report_lof_states = evaluate(preds=predictions["LOF_states"], labels=clean_labels)
        report_lof_gates = evaluate(preds=predictions["LOF_gates"], labels=clean_labels)
        report_lof5_states = evaluate(preds=predictions["LOF_5_states"], labels=clean_labels[3:-2])
        report_lof10_states = evaluate(preds=predictions["LOF_10_states"], labels=clean_labels[6:-5])
        report_lof5_gates = evaluate(preds=predictions["LOF_5_gates"], labels=clean_labels[3:-2])
        report_lof10_gates = evaluate(preds=predictions["LOF_10_gates"], labels=clean_labels[6:-5])
        report_lof5_ts = evaluate(preds=predictions["LOF_5_TS"], labels=clean_labels[3:-2])
        report_lof10_ts = evaluate(preds=predictions["LOF_10_TS"], labels=clean_labels[6:-5])

        report_isof_ts = evaluate(preds=predictions["ISOF_TS"], labels=clean_labels)
        report_isof_states = evaluate(preds=predictions["ISOF_states"], labels=clean_labels)
        report_isof_gates = evaluate(preds=predictions["ISOF_gates"], labels=clean_labels)
        report_isof5_ts = evaluate(preds=predictions["ISOF_5_TS"], labels=clean_labels[3:-2])
        report_isof10_ts = evaluate(preds=predictions["ISOF_10_TS"], labels=clean_labels[6:-5])
        report_isof5_states = evaluate(preds=predictions["ISOF_5_states"], labels=clean_labels[3:-2])
        report_isof10_states = evaluate(preds=predictions["ISOF_10_states"], labels=clean_labels[6:-5])
        report_isof5_gates = evaluate(preds=predictions["ISOF_5_gates"], labels=clean_labels[3:-2])
        report_isof10_gates = evaluate(preds=predictions["ISOF_10_gates"], labels=clean_labels[6:-5])

        all_reports = {"rnn": report_rnn,
                       "lof_ts": report_lof,
                       "lof_states": report_lof_states,
                       "lof_gates": report_lof_gates,
                       "lof_5_states": report_lof5_states,
                       "lof_10_states": report_lof10_states,
                       "lof_5_gates": report_lof5_gates,
                       "lof_10_gates": report_lof10_gates,
                       "lof_5_ts": report_lof5_ts,
                       "lof_10_ts": report_lof10_ts,
                       "isof_states": report_isof_states,
                       "isof_gates": report_isof_gates,
                       "isof_5_states": report_isof5_states,
                       "isof_10_states": report_isof10_states,
                       "isof_5_gates": report_isof5_gates,
                       "isof_10_gates": report_isof10_gates,
                       "isof_ts": report_isof_ts,
                       "isof5_ts": report_isof5_ts,
                       "isof10_ts": report_isof10_ts}
        printEval(all_reports)
        return all_reports



    else:
        raise NotImplementedError

def evaluate(preds, labels):
    report = classification_report(y_true=labels,
                                   y_pred=preds,
                                   target_names=["No Outlier", "Outlier"],
                                   output_dict=True)

    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    N_pos = np.sum(preds == 1)
    N_neg = np.sum(preds == 0)
    report["TN"] = tn / N_neg if N_neg != 0 else 0
    report["FP"] = fp / N_pos if N_pos != 0 else 0
    report["FN"] = fn / N_neg if N_neg != 0 else 0
    report["TP"] = tp / N_pos if N_pos != 0 else 0







    return report


def completeEvaluation(lstm, minimalrnn, mgu):
    minimalrnn_df = pd.DataFrame(data=np.zeros(shape=(len(minimalrnn[0]), len(minimalrnn[0]))),
                                 index=minimalrnn[0].keys(),
                                 columns=minimalrnn[0].keys())
    n_iter = len(minimalrnn)
    for i in range(n_iter):
        for m1 in minimalrnn[0].keys():
            for m2 in minimalrnn[0].keys():
                m1_f1_score = minimalrnn[i][m1]["Outlier"]["f1-score"]
                m2_f1_score = minimalrnn[i][m2]["Outlier"]["f1-score"]
                if m1_f1_score > m2_f1_score:
                    minimalrnn_df.loc[m1, m2] += 1

    minimalrnn_df.to_csv("results_minimalrnn.csv")

    lstm_df = pd.DataFrame(data=np.zeros(shape=(len(lstm[0]), len(lstm[0]))),
                                 index=lstm[0].keys(),
                                 columns=lstm[0].keys())
    n_iter = len(lstm)
    for i in range(n_iter):
        for m1 in lstm[0].keys():
            for m2 in lstm[0].keys():
                m1_f1_score = lstm[i][m1]["Outlier"]["f1-score"]
                m2_f1_score = lstm[i][m2]["Outlier"]["f1-score"]
                if m1_f1_score > m2_f1_score:
                    lstm_df.loc[m1, m2] += 1

    lstm_df.to_csv("results_lstm.csv")

    mgu_df = pd.DataFrame(data=np.zeros(shape=(len(mgu[0]), len(mgu[0]))),
                                 index=mgu[0].keys(),
                                 columns=mgu[0].keys())
    n_iter = len(mgu)
    for i in range(n_iter):
        for m1 in mgu[0].keys():
            for m2 in mgu[0].keys():
                m1_f1_score = mgu[i][m1]["Outlier"]["f1-score"]
                m2_f1_score = mgu[i][m2]["Outlier"]["f1-score"]
                if m1_f1_score > m2_f1_score:
                    mgu_df.loc[m1, m2] += 1

    mgu_df.to_csv("results_mgu.csv")


    n_iter = len(lstm)
    """minimalrnn_df = pd.DataFrame(data=np.zeros(shape=(len(minimalrnn[0]), 2)),
                                 index=minimalrnn[0].keys(),
                                 columns=["No Outlier", "Outlier"])
    lstm_df = pd.DataFrame(data=np.zeros(shape=(len(lstm[0]), 2)),
                           index=lstm[0].keys(),
                           columns=["No Outlier", "Outlier"])
    mgu_df = pd.DataFrame(data=np.zeros(shape=(len(mgu[0]), 2)),
                          index=mgu[0].keys(),
                          columns=["No Outlier", "Outlier"])

    for i in range(n_iter):
        best_method_outl, best_method_no_outl = None, None
        curr_best_outl, curr_best_no_outl = -np.inf, -np.inf
        for method in minimalrnn[i]:
            no_outl_f1 = minimalrnn[i][method]["No Outlier"]["f1-score"]
            outl_f1 = minimalrnn[i][method]["Outlier"]["f1-score"]

            if outl_f1 > curr_best_outl:
                best_method_outl = method
                curr_best_outl = outl_f1
            if no_outl_f1 > curr_best_no_outl:
                best_method_no_outl = method
                curr_best_no_outl = no_outl_f1
        minimalrnn_df.loc[best_method_outl, "Outlier"] += 1
        minimalrnn_df.loc[best_method_no_outl, "No Outlier"] += 1

        best_method_outl, best_method_no_outl = None, None
        curr_best_outl, curr_best_no_outl = -np.inf, -np.inf
        for method in lstm[i]:
            no_outl_f1 = lstm[i][method]["No Outlier"]["f1-score"]
            outl_f1 = lstm[i][method]["Outlier"]["f1-score"]

            if outl_f1 > curr_best_outl:
                best_method_outl = method
                curr_best_outl = outl_f1
            if no_outl_f1 > curr_best_no_outl:
                best_method_no_outl = method
                curr_best_no_outl = no_outl_f1
        lstm_df.loc[best_method_outl, "Outlier"] += 1
        lstm_df.loc[best_method_no_outl, "No Outlier"] += 1

        best_method_outl, best_method_no_outl = None, None
        curr_best_outl, curr_best_no_outl = -np.inf, -np.inf
        for method in mgu[i]:
            no_outl_f1 = mgu[i][method]["No Outlier"]["f1-score"]
            outl_f1 = mgu[i][method]["Outlier"]["f1-score"]

            if outl_f1 > curr_best_outl:
                best_method_outl = method
                curr_best_outl = outl_f1
            if no_outl_f1 > curr_best_no_outl:
                best_method_no_outl = method
                curr_best_no_outl = no_outl_f1
        mgu_df.loc[best_method_outl, "Outlier"] += 1
        mgu_df.loc[best_method_no_outl, "No Outlier"] += 1

    minimalrnn_df /= n_iter
    lstm_df /= n_iter
    mgu_df /= n_iter

    print("=== MINIMALRNN RESULTS ===")
    print(minimalrnn_df)
    print()

    print("=== LSTM RESULTS ===")
    print(lstm_df)
    print()

    print("=== MGU RESULTS ===")
    print(mgu_df)"""






if __name__ == "__main__":
    path = "modifiedDatasets/sunspot/data5/"
    train_ts = np.load(path + "train.npy")
    test_ts = np.load(path + "test.npy")
    outl_ts = np.load(path + "outl.npy")
    labels = np.load(path + "labels.npy")

    initializeExperiment(type="MinimalRNN/sunspot")
    logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
    stdout = sys.stdout
    sys.stdout = logfile
    res = runExperiment(type="MinimalRNN", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
    sys.stdout = stdout
    logfile.close()



    """lstm_results, mgu_results, minimalrnn_results = [], [], []
    for i in range(10):
        #path = "modifiedDatasets/ecg1/data" + str(i) + "/"
        path = "modifiedDatasets/sunspot/data" + str(i) + "/"
        train_ts = np.load(path + "train.npy")
        test_ts = np.load(path + "test.npy")
        outl_ts = np.load(path + "outl.npy")
        labels = np.load(path + "labels.npy")


        initializeExperiment(type="LSTM/sunspot")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile
        res = runExperiment(type="LSTM", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        lstm_results.append(res)
        sys.stdout = stdout
        logfile.close()

        initializeExperiment(type="MinimalRNN/sunspot")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile
        res = runExperiment(type="MinimalRNN", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        minimalrnn_results.append(res)
        sys.stdout = stdout
        logfile.close()

        initializeExperiment(type="MGU/sunspot")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile
        res = runExperiment(type="MGU", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        mgu_results.append(res)
        sys.stdout = stdout
        logfile.close()

    completeEvaluation(lstm=lstm_results, minimalrnn=minimalrnn_results, mgu=mgu_results)"""




    """lstm_results, mgu_results, minimalrnn_results = [], [], []
    ids = [i for i in range(36)]
    for id in ids:
        ds = "global/data" + str(id)
        dataset, info, img = SyntheticData.load(dataset=ds)
        train_ts, test_ts, outl_ts, labels = dataset

        # LSTM
        initializeExperiment(type="LSTM/synth_global")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile

        res = runExperiment(type="LSTM", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        lstm_results.append(res)

        sys.stdout = stdout
        logfile.close()

        # MinimalRNN
        initializeExperiment(type="MinimalRNN/synth_global")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile

        res = runExperiment(type="MinimalRNN", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        minimalrnn_results.append(res)

        sys.stdout = stdout
        logfile.close()

        # MRU
        initializeExperiment(type="MGU/synth_global")
        logfile = open(PLOTS_COMMON_NAME + "/log.txt", "w")
        stdout = sys.stdout
        sys.stdout = logfile

        res = runExperiment(type="MGU", train_ts=train_ts, test_ts=test_ts, outl_ts=outl_ts, labels=labels)
        mgu_results.append(res)

        sys.stdout = stdout
        logfile.close()

    completeEvaluation(lstm=lstm_results, minimalrnn=minimalrnn_results, mgu=mgu_results)"""