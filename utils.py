import numpy as np
import sys
import pandas as pd
from random import random, randint
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def array_softmax(Q, T):
    """Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float."""
    P = np.zeros(Q.shape)
    TdQ = -T * (Q[:, 0] - Q[:, 1])
    TdQ[
        TdQ > LOG_MAX_FLOAT
    ] = LOG_MAX_FLOAT  # Protection against overflow in exponential.
    P[:, 0] = 1.0 / (1.0 + np.exp(TdQ))
    P[:, 1] = 1.0 - P[:, 0]
    return P


def protected_log(x):
    "Return log of x protected against giving -inf for very small values of x."
    return np.log(((1e-200) / 2) + (1 - (1e-200)) * x)


def choose(P):
    "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
    return sum(np.cumsum(P) < random())


def softmax(Q, T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[
        QT > LOG_MAX_FLOAT
    ] = LOG_MAX_FLOAT  # Protection against overflow in exponential.
    expQT = np.exp(QT)
    return expQT / expQT.sum()


def latent_state_simulate(params, r_states, n_trials, rprob):

    p_r, p_lapse = params
    good_prob = 0.8

    p_1 = 0.5  # Probability world is in state 1.
    p_all = np.zeros(n_trials + 1)
    p_all[0] = p_1
    p_o_1 = np.array(
        [
            [
                good_prob,
                1 - good_prob,
            ],  # Probability of observed outcome given world in state 1.
            [1 - good_prob, good_prob],
        ]
    )  # Indicies:  p_o_1[second_step, outcome]

    p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

    choices, second_steps, outcomes = (
        np.zeros(n_trials, int),
        np.zeros(n_trials, int),
        np.zeros(n_trials, int),
    )

    for i in range(n_trials):

        # Generate trial events.
        c = s = int((p_1 > 0.5) == (random() > p_lapse))
        o = int(random() <= rprob) if c == r_states[i] else int(random() > rprob)

        # Bayesian update of state probabilties given observed outcome.
        p_1 = p_o_1[s, o] * p_1 / (p_o_1[s, o] * p_1 + p_o_0[s, o] * (1 - p_1))
        # Update of state probabilities due to possibility of block reversal.
        p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)
        p_all[i + 1] = p_1
        choices[i], second_steps[i], outcomes[i] = (c, s, o) # TODO： why c == s ?

    return choices, second_steps, outcomes, p_all


def latent_state_session_likelihood(params, results, return_q=False):
    n_trials = results.shape[0]
    # Unpack trial events.
    choices, second_steps, outcomes = (
        results["choices"].astype(int),
        results["second_steps"].astype(int),
        results["outcomes"].astype(int),
    )

    # Unpack parameters.
    p_r, p_lapse = params
    good_prob = 0.8

    p_o_1 = np.array(
        [
            [
                good_prob,
                1 - good_prob,
            ],  # Probability of observed outcome given world in state 1.
            [1 - good_prob, good_prob],
        ]
    )  # Indicies:  p_o_1[second_step, outcome]

    p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

    p_1 = np.zeros(n_trials + 1)  # Probability world is in state 1.
    p_1[0] = 0.5

    for i, (c, s, o) in enumerate(
        zip(choices, second_steps, outcomes)
    ):  # loop over trials.

        # Bayesian update of state probabilties given observed outcome.
        p_1[i + 1] = (
            p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
        ) #TODO: why use p_o_1?
        # Update of state probabilities due to possibility of block reversal.
        p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

    # Evaluate choice probabilities and likelihood.
    choice_probs = np.zeros([n_trials + 1, 2])
    choice_probs[:, 1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
    choice_probs[:, 0] = 1 - choice_probs[:, 1]
    trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
    session_log_likelihood = np.sum(trial_log_likelihood)

    if return_q:
        return -session_log_likelihood, choice_probs
    else:
        return -session_log_likelihood


def RL_session_likelihood(params, results, return_q=False):
    n_trials = results.shape[0]
    # Unpack trial events.
    choices, second_steps, outcomes = (
        results["choices"].astype(int),
        results["second_steps"].astype(int),
        results["outcomes"].astype(int),
    )
    # Unpack parameters.
    alpha_td, b1, b2, b3 = params
    alpha_qlearning = b4 = 0
    #     alpha_qlearning, b2, b3, b4 = params
    #     alpha_td = b1 = 0
    #     alpha_td, alpha_qlearning, b1, b2, b3, b4 = params
    iTemp = 1

    # Variables.
    Q_td = np.zeros([2, 1, 2])  # Indicies: action, prev. second step., prev outcome
    Q_qlearning = np.zeros([2, 1, 2])

    Q_td2 = np.zeros([n_trials, 2])  # Active action values.
    Q_qlearning2 = np.zeros([n_trials, 2])
    Q_bias = np.ones([n_trials, 2])  # bias agent.

    ps = 0  # previous second step.
    #         po = randint(0, 1)  # previous outcome.
    po = 0
    for i, (c, s, o) in enumerate(
        zip(choices, second_steps, outcomes)
    ):  # loop over trials.

        Q_td2[i, 0] = Q_td[0, ps, po]
        Q_td2[i, 1] = Q_td[1, ps, po]

        Q_qlearning2[i, 0] = Q_qlearning[0, ps, po]
        Q_qlearning2[i, 1] = Q_qlearning[1, ps, po]

        # update action values.
        Q_td[c, ps, po] = (1.0 - alpha_td) * Q_td[c, ps, po] + alpha_td * o
        Q_qlearning[c, ps, po] = Q_qlearning[c, ps, po] + alpha_qlearning * (
            o - Q_qlearning[c, ps, po]
        ) #TODO Q(1) here? why alpha_learning is set to 0 here? why not use alpha_td here?
        # Q_qlearning[c, ps, po] = Q_qlearning[c, ps, po] + alpha_td * (
        #     o - Q_qlearning[c, ps, po]
        # )

        ps, po = s, o

    Q_psv = np.array(
        [
            np.array([1, 0]) if i == 0 else np.array([0, 1])
            for i in np.roll(results["choices"], 1)
        ]
    )
    Q_psv[0, :] = np.array([0, 0])
    for i in range(n_trials):
        Q_bias[i, randint(0, 1)] = -1

    # Evaluate choice probabilities and likelihood.
    Q = b1 * Q_td2 + b2 * Q_psv + b3 * Q_bias + b4 * Q_qlearning2
    choice_probs = array_softmax(Q, iTemp)
    trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
    session_log_likelihood = np.sum(trial_log_likelihood)
    if return_q:
        return -session_log_likelihood, Q
    else:
        return -session_log_likelihood


def RL_simulate(params, r_states, n_trials, rprob):
    alpha_td, b1, b2, b3 = params
    alpha_qlearning = b4 = 0
    #     alpha_qlearning, b2, b3, b4 = params
    #     alpha_td = b1 = 0
    #     alpha_td, alpha_qlearning, b1, b2, b3, b4 = params
    iTemp = 1

    Q_td = np.zeros([2, 1, 2])  # Indicies: action, prev. second step., prev outcome
    Q_qlearning = np.zeros([2, 1, 2])

    Q_td2 = np.zeros([n_trials, 2])
    Q_qlearning2 = np.zeros([n_trials, 2])
    choices, second_steps, outcomes = (
        np.zeros(n_trials, int),
        np.zeros(n_trials, int),
        np.zeros(n_trials, int),
    )

    ps = 0  # previous second step.
    #         po = randint(0, 1)  # previous outcome.
    po = 0

    for i in range(n_trials):
        #             set_trace()
        Q_td2[i, 0] = Q_td[0, ps, po]
        Q_td2[i, 1] = Q_td[1, ps, po]

        Q_qlearning2[i, 0] = Q_qlearning[0, ps, po]
        Q_qlearning2[i, 1] = Q_qlearning[1, ps, po]
        # Generate trial events.
        if i == 0:
            c = choose(
                softmax(
                    (
                        b1 * Q_td[:, ps, po]
                        + b2 * randint(0, 1)
                        + b3 * randint(-1, 1)
                        + b4 * Q_qlearning[:, ps, po]
                    ),
                    iTemp,
                )
            )
        else:
            c = choose(
                softmax(
                    (
                        b1 * Q_td[:, ps, po]
                        + b2 * int(c == choices[i - 1])
                        + b3 * randint(-1, 1)
                        + b4 * Q_qlearning[:, ps, po]
                    ),
                    iTemp,
                )
            )
        s = 0
        o = int(random() <= rprob) if c == r_states[i] else int(random() > rprob)

        # update action values.
        Q_td[c, ps, po] = (1.0 - alpha_td) * Q_td[c, ps, po] + alpha_td * o
        Q_qlearning[c, ps, po] = Q_qlearning[c, ps, po] + alpha_qlearning * (
            o - Q_qlearning[c, ps, po]
        )
        ps, po = s, o

        choices[i], second_steps[i], outcomes[i] = (c, s, o)

    return choices, second_steps, outcomes, Q_td2, Q_qlearning2


def before_after_chg_accuracy(results, choices_pred):
    results = results.assign(choices_pred=choices_pred)
    """where states change"""
    change_index = (
        (results.r_state != results.r_state.shift())
        .where(lambda x: x == True)
        .dropna()
        .index[1:]
    )

    """get 20 indices before and after change index"""
    buffer_length = 20
    before_after_chgs = pd.DataFrame(
        [
            [
                results.loc[i, "r_state"],
                results.loc[
                    i - buffer_length : i + buffer_length, ["choices", "choices_pred"]
                ]
                .apply(lambda x: x.values[0] == x.values[1], 1)
                .values.ravel(),
            ]
            for i in change_index
        ],
        columns=["to_state", "accuracy"],
    )

    before_after_chgs = before_after_chgs[
        before_after_chgs.accuracy.apply(len) == 1 + buffer_length * 2
    ]
    """plot"""
    plt.figure(figsize=(15, 3))
    plt.plot(np.vstack(before_after_chgs.accuracy).mean(0))
    plt.xticks(
        ticks=range(1 + buffer_length * 2),
        labels=np.arange(-buffer_length, buffer_length + 1),
    )
    plt.xlabel("before/after change points (0 is the start of the later state)")
    plt.ylabel("average accuracy")
    plt.title(label="before and after state transition accuracy")
    plt.show()


def do_logistic_regression(L, rwds, actions, ax, method="rwd_choice"):
    if method == "rwd_choice":
        data = (
            pd.concat(
                [
                    pd.Series(rwds * actions).shift(i).rename("last" + str(i))
                    for i in range(1, L)
                ],
                1,
            )
            .assign(y=actions)
            .dropna()
        )
    if method == "rwd":
        data = (
            pd.concat(
                [pd.Series(rwds).shift(i).rename("last" + str(i)) for i in range(1, L)],
                1,
            )
            .assign(y=actions)
            .dropna()
        )
    if method == "choice":
        data = (
            pd.concat(
                [
                    pd.Series(actions).shift(i).rename("last" + str(i))
                    for i in range(1, L)
                ],
                1,
            )
            .assign(y=actions)
            .dropna()
        )

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("{} accuracy:".format(method), accuracy_score(y_test, pred))
    pd.DataFrame(model.coef_, columns=X.columns).T.rename(columns={0: "weight"}).plot(
        ax=ax, marker="o"
    )


LOG_MAX_FLOAT = np.log(sys.float_info.max / 2.1)
