import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def _CPD(X, y):  # 这里的X的shape是273*6，y的shape是273*1800
    """Evaluate coefficient of partial determination for each predictor in X"""
    ols = LinearRegression()  # 建立一个linear regression的拟合object
    #     set_trace()
    ols.fit(X, y)  # 拟合X和y
    sse = np.sum(
        (ols.predict(X) - y) ** 2, axis=0
    )  # 用X预测y，然后和真实的y做残差平方和，sse的shape是1800*1
    cpd = np.zeros([y.shape[1], X.shape[1]])  # 建立一个空的matrix，等等要存储结果，shape是1800*6
    for i in range(X.shape[1]):  # i的loop从0-5，一共6个数
        X_i = np.delete(X, i, axis=1)  # 这里X_i表示"X这个矩阵去掉第i列",所以X_i的shape永远是273*5
        ols.fit(X_i, y)  # 用去掉一列的X_i和y进行拟合
        sse_X_i = np.sum(
            (ols.predict(X_i) - y) ** 2, axis=0
        )  # 用X_i预测y，然后和真实的y做残差平方和，sse_X_i的shape是1800*1
        cpd[:, i] = (
            sse_X_i - sse
        ) / sse_X_i  # 算两种残差的差异，用比例的方式表达，同时把结果存储到cpd这个matrix里面（放到第i列里面）
    return cpd

# Read data
with open("xieyang_neurondata.pkl", "rb") as f:
    pkl = pickle.load(f)
df = pd.read_csv("xieyang_behaviordata.csv")



all_type_avgs = []
for i in pkl:
    file_name, indexes = i["file"], i["trial_selected"]
    indexes = indexes[0].astype(int) - 1  # matlab index to python index
    X_choice = (
        df[
            df.file.apply(lambda x: "_".join(x.split("_")[:2]))
            == "_".join(file_name.split("_")[:2])
        ]
        .iloc[indexes, :][["rwd_fed", "c_cue"]]
        .rename(columns={"rwd_fed": "reward", "c_cue": "choice"})
        .replace({-1: 0})
    )
    """masks为四种type的bool mask"""
    masks = (
        X_choice.reset_index()
        .groupby(["reward", "choice"])
        .apply(lambda x: list(x.index))
        .values
    )
    """根据每一种type，在trial的维度上（维度0）上average，同时concatenate"""
    type_avgs = np.concatenate(
        [np.mean(i["Fd2"][m, ::], 0, keepdims=True) for m in masks], 0
    )  # n_type * n_timepoints * n_neurons
    all_type_avgs.append(type_avgs)

print()
combine_neuron_avgs = np.concatenate(
    all_type_avgs, 2
)  # n_type * n_timepoints * all n_neurons
combine_neuron_avgs = combine_neuron_avgs - np.mean(
    combine_neuron_avgs, 0
)  # n_type * n_timepoints * all n_neurons
combine_neuron_avgs = combine_neuron_avgs.swapaxes(
    2, 1
)  # n_type * all n_neurons * n_timepoints
X = np.hstack(
    [caa for caa in combine_neuron_avgs]
)  # [all n_neurons, n_timepoints * n_type]

pca = PCA(n_components=12)
pca.fit(X.T)

#%%

colors = ["b", "r", "c", "orange"]
for i, caa in enumerate(combine_neuron_avgs):
    # 3D plot
    traj_x = caa.T @ pca.components_[0, :]
    traj_y = caa.T @ pca.components_[1, :]
    plt.plot(traj_x, traj_y, color=colors[i])
    for j, sn, m in zip(range(3), [0, 10, 25], ["$S$", "$O$", "$E$"]):
        plt.scatter(traj_x[sn], traj_y[sn], color=colors[i], marker=m, s=80)
plt.legend(
    (
        X_choice.reset_index()
        .groupby(["reward", "choice"])
        .apply(lambda x: list(x.index))
        .reset_index()
    )[["reward", "choice"]].values
)
plt.show()