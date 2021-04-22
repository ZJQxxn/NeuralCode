import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from itertools import groupby
from scipy.optimize import minimize, brute, fmin
from utils import *


def _readData():
    df = (
        pd.read_csv("xieyang_behaviordata.csv")
            .rename(columns={"c_cue": "choices", "rwd_fed": "outcomes"})
            .assign(second_steps=0)
    )
    results = df[df.file == "M150241_session6_bhv_ana.mat"].replace(-1, 0)
    n_trials = results.shape[0]


def fitting():
    """fit the parameters"""
    initial = brute(
        session_likelihood,
        ranges=(
            slice(0, 1, 0.1),
            #         slice(0, 1, 0.1),
            #         slice(0, 10, 1),
            slice(0, 10, 1),
            slice(0, 10, 1),
            slice(0, 10, 1),
        ),
        finish=fmin,
        workers=-1,
    )
    res = minimize(
        session_likelihood,
        x0=initial,
        method="SLSQP",
        bounds=[(0, 1), (0, None), (0, None), (0, None)],
        options={"eps": 0.1, "disp": True, "ftol": 1e-10},
    )
    print("fitted:", res.x)

    """check likelihood and Q"""
    likelihood, Q = session_likelihood(res.x, True)
    print("likelihood:", likelihood)




if __name__ == '__main__':

print("trial length:", n_trials)
print("current reward probability:", results.rprob.unique()[0])
print(
    "reward states and each block length:\n",
    pd.DataFrame(
        [[idx, len(list(item))] for idx, item in groupby(results.r_state)],
        columns=["r_state", "state_length"],
    ),
)

"""fit the parameters"""
initial = brute(
    session_likelihood,
    ranges=(
        slice(0, 1, 0.1),
        #         slice(0, 1, 0.1),
        #         slice(0, 10, 1),
        slice(0, 10, 1),
        slice(0, 10, 1),
        slice(0, 10, 1),
    ),
    finish=fmin,
    workers=-1,
)
res = minimize(
    session_likelihood,
    x0=initial,
    method="SLSQP",
    bounds=[(0, 1), (0, None), (0, None), (0, None)],
    options={"eps": 0.1, "disp": True, "ftol": 1e-10},
)
print("fitted:", res.x)

"""check likelihood and Q"""
likelihood, Q = session_likelihood(res.x, True)
print("likelihood:", likelihood)

plt.figure(figsize=(15, 3))
plt.plot(Q[:200])
plt.title("First 200 Q changes")

"""predict choice using fitted parameters"""
(
    choices_pred,
    second_steps_pred,
    outcomes_pred,
    Q_td2_pred,
    Q_qlearning2_pred,
) = simulate(res.x, results.r_state.values, n_trials, results.rprob.values[0])

print("prediction accuracy:", (choices_pred == results.choices).mean())

before_after_chg_accuracy(results, choices_pred)

#%%

fig, ax = plt.subplots(figsize=(13, 3))
ax.set_title("prediction")
for i in ["rwd_choice", "rwd", "choice"]:
    do_logistic_regression(11, outcomes_pred, choices_pred, ax, i)
plt.legend(["rwd_choice", "rwd", "choice"])

fig, ax = plt.subplots(figsize=(13, 3))
ax.set_title("actual")
for i in ["rwd_choice", "rwd", "choice"]:
    do_logistic_regression(11, results.outcomes.values, results.choices.values, ax, i)
plt.legend(["rwd_choice", "rwd", "choice"])
plt.show()