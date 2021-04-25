import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sbn
import sys
from itertools import groupby
from scipy.optimize import minimize, brute, fmin
from utils import *


def _readBehavioralData():
    df = (
        pd.read_csv("./data/xieyang_behaviordata.csv")
            .rename(columns={"c_cue": "choices", "rwd_fed": "outcomes"})
            .assign(second_steps=0)
    )
    filename = df.file.unique()
    print("Num of files : ", len(filename))
    file = "M150241_session6_bhv_ana.mat"
    # file = filename[0]
    print("* Chosen File * ", file)
    results = df[df.file == file].replace(-1, 0) # TODO: change file
    return df, results


def _readNueralData():
    with open("./data/xieyang_neurondata.pkl", "rb") as f:
        pkl = pickle.load(f)
    return pkl


def _fitting(initial, results, method):
    if method == "RL":
        session_likelihood = RL_session_likelihood
    elif method == "latent":
        session_likelihood = latent_state_session_likelihood
    else:
        raise ValueError("Undefined method {}!".format(method))
    func = lambda params: session_likelihood(
        params,
        results,
        return_q=False
    )
    res = minimize(
        func,
        x0=initial,
        method="SLSQP",
        bounds=[(0, 1), (0, None), (0, None), (0, None)] if method == "RL" else [(0, 1), (0, 1),],
        options={"eps": 0.1, "disp": True, "ftol": 1e-10},
    )
    return res


def _prediction(par, results, method):
    n_trials = results.shape[0]
    # predict choice using fitted parameters
    if method == "RL":
        simulate = RL_simulate
        (
            choices_pred,
            second_steps_pred,
            outcomes_pred,
            Q_td2_pred,
            Q_qlearning2_pred,
        ) = simulate(par, results.r_state.values, n_trials, results.rprob.values[0])
        print("Prediction accuracy:", np.nanmean(choices_pred == results.choices))
        print("-"*50)
    elif method == "latent":
        simulate = latent_state_simulate
        (
            choices_pred,
            second_steps_pred,
            outcomes_pred,
            p_pred
        ) = simulate(par, results.r_state.values, n_trials, results.rprob.values[0])
        print("Prediction accuracy:", np.nanmean(choices_pred == results.choices))
        print("-"*50)
        plt.figure(figsize=(13, 3))
        plt.title("Predicted p")
        plt.plot(p_pred)
        plt.show()
        plt.close()
    else:
        raise ValueError("Undefined method {}!".format(method))



    before_after_chg_accuracy(results, choices_pred)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(2, 1 , 1)
    ax.set_title("Prediction")
    for i in ["rwd_choice", "rwd", "choice"]:
        do_logistic_regression(11, outcomes_pred, choices_pred, ax, i)
    plt.legend(["rwd_choice", "rwd", "choice"])
    #
    # fig, ax = plt.subplots(figsize=(13, 3))
    ax = fig.add_subplot(2, 1, 2)
    ax.set_title("Actual")
    for i in ["rwd_choice", "rwd", "choice"]:
        do_logistic_regression(11, results.outcomes.values, results.choices.values, ax, i)
    plt.legend(["rwd_choice", "rwd", "choice"])
    plt.show()


def _initialization(results, method):
    if method == "RL":
        session_likelihood = RL_session_likelihood
        func = lambda params: session_likelihood(
            params,
            results,
            return_q=False
        )
        initial = brute(
            func,
            ranges=(
                slice(0, 1, 0.1),
                slice(0, 10, 1),
                slice(0, 10, 1),
                slice(0, 10, 1),
                # slice(0, 1, 0.2),
                # slice(0, 10, 4),
                # slice(0, 10, 4),
                # slice(0, 10, 4),
            ),
            finish=fmin,
            workers=-1,
        )
    elif method == "latent":
        session_likelihood = latent_state_session_likelihood
        func = lambda params: session_likelihood(
            params,
            results,
            return_q=False
        )
        initial = brute(
            func,
            ranges=(
                slice(0, 1, 0.1),
                slice(0, 1, 0.1)
            ),
            finish=fmin,
            workers=-1,
        )
    else:
        raise ValueError("Undefined method {}!".format(method))

    return initial


def RLModelFitting(config):
    import warnings
    warnings.filterwarnings("ignore")

    # Configuration
    method = config["method"]  # "RL" or "latent"
    need_init = config["need_init"]
    need_fit = config["need_fit"]
    par = config["par"]
    print("=" * 50)
    print("Reading data...")
    df, results = _readBehavioralData()
    n_trials = results.shape[0]
    print("Trial length:", n_trials)
    print("Current reward probability:", results.rprob.unique()[0])

    tmp = pd.DataFrame(
        [[idx, len(list(item))] for idx, item in groupby(results.r_state)],
        columns=["r_state", "state_length"],
    )
    sbn.histplot(tmp.state_length, bins=[40, 50, 60, 70, 80, 90, 100])
    plt.title("State Length")
    plt.show()
    # ===========================
    print("-" * 50)
    print("RL fitting...")
    print("Start initializing parameters...")
    if need_init is True:
        initial = _initialization(results, method)
    else:
        initial = config["initial"]
    # Model fitting
    if need_fit is True:
        print("Initial parameters : ", initial)
        res = _fitting(initial, results, method)
        par = res.x
        print("Fitted parameters:", res.x)
    else:
        par = par
        print("Specified parameters:", par)
    if method == "RL":
        session_likelihood = RL_session_likelihood
    elif method == "latent":
        session_likelihood = latent_state_session_likelihood
    else:
        raise ValueError("Undefined method {}!".format(method))
    # Model testing
    likelihood, Q = session_likelihood(par, results, return_q=True)
    print("Likelihood:", likelihood)
    plt.figure(figsize=(15, 3))
    plt.plot(Q[:200])
    plt.title("First 200 Q changes")
    plt.show()
    # =====================================
    print("-" * 50)
    print("Prediction:")
    _prediction(par, results, method)
    print("=" * 50)


if __name__ == '__main__':
    # Configurations
    config = {
        "method": "latent",
        "need_init": False,
        "need_fit": False,
        # -------------------------------
        #           FOR RL
        # "par":[0.60039157, 2.13713513, 0.88885334, 3.37012926],
        # "initial": [0.1, 1, 1, 1],
        # -------------------------------
        #           FOR LATENT
        "par": [0.00794711, 0.16998064],
        "initial": [0.1, 0.1]
    }

    # Model fitting
    RLModelFitting(config)






