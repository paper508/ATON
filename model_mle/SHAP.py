import shap
import math
import random
import sklearn
import numpy as np
from model_mle.clf_utils import classifiers


class SHAP:
    def __init__(self, kernel="rbf", n_sample=100, threshold=0.8):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.ano_idx = None

        self.kernel = kernel
        self.threshold = threshold
        self.n_sample = n_sample
        self.dim = None
        return

    def fit(self, x, y):

        self.dim = x.shape[1]

        # metric_lst = []
        # clf_lst = []
        # for model in classifiers.keys():
        #     clf = classifiers[model]
        #     clf.fit(x, y)
        #     y_pred = clf.predict(x)
        #     clf_lst.append(clf)
        #     metric_lst.append(sklearn.metrics.f1_score(y, y_pred))
        # choose_idx = int(np.argmax(metric_lst))
        # clf = clf_lst[choose_idx]
        # print("Choosing Clf: [%s]" % list(classifiers.keys())[choose_idx])

        clf = sklearn.svm.SVC(kernel=self.kernel, probability=True)
        clf.fit(x, y)

        y_pred = clf.predict(x)
        print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y, y_pred)))

        self.ano_idx = np.where(y == 1)[0]

        # use Kernel SHAP to explain test set predictions
        x_kmean = shap.kmeans(x, self.n_sample)
        explainer = shap.KernelExplainer(clf.predict_proba, x_kmean, link="logit")

        anomaly_shap_values = explainer.shap_values(x[self.ano_idx], nsamples="auto")
        anomaly_shap_values = anomaly_shap_values[1]
        return anomaly_shap_values

    def weight2subspace(self, weight, r=0.7, num=-1):
        threshold = r * np.sum(weight)
        tmp_s = 0
        exp_subspace = []
        sorted_idx1 = np.argsort(weight)
        sorted_idx = [sorted_idx1[self.dim - i -1] for i in range(self.dim)]
        if num != -1:
            exp_subspace = sorted_idx[:num]
            exp_subspace = list(np.sort(exp_subspace))
            return exp_subspace

        for idx in sorted_idx:
            tmp_s += weight[idx]
            exp_subspace.append(idx)
            if tmp_s >= threshold:
                break
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def weight2subspace_pn(self, weight):
        exp_subspace = []
        for i in range(len(weight)):
            if weight[i] > 0:
                exp_subspace.append(i)
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def get_exp_subspace(self, fea_weight_lst, w2s_ratio, real_exp_len=None):
        exp_subspace_lst = []
        for ii, idx in enumerate(self.ano_idx):
            fea_weight = fea_weight_lst[ii]
            if w2s_ratio == "real_len":
                exp_subspace_lst.append(self.weight2subspace(fea_weight, num=real_exp_len[ii]))
            elif w2s_ratio == "auto":
                r = math.sqrt(2 / self.dim)
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=r))
            elif w2s_ratio == "pn":
                exp_subspace_lst.append(self.weight2subspace_pn(fea_weight))
            else:
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=w2s_ratio))
        return exp_subspace_lst



    # -------- diting method ---------- #
    def fit_diting(self, df, chunk=-1):
        X, y, columns_name = self.diting_pre(df)
        full = np.arange(len(X))
        if full > 2000:
            train_idx = random.sample(full, 2000)
        else:
            train_idx = full
        X_train = X[train_idx]
        y_train = y[train_idx]
        explainer = self.diting_train(X_train, y_train)

        fail_idx = np.where(y == 1)[0]
        fail_x = X[fail_idx]
        shap_values = self.diting_test(fail_x, explainer)[1]

        max_value = np.max(shap_values, axis=1)
        max_value = np.array([(a - np.min(max_value))/(np.max(max_value)-np.min(max_value)) for a in max_value])
        exp_idx = np.argmax(shap_values, axis=1)
        exp_col_name = [columns_name[a] for a in exp_idx]
        predict_df = df[['trace_id', 'device_id', 'cluster_id', 'span_name', 'label']]
        predict_df = predict_df.loc[predict_df["label"] == 1]
        predict_df["reason"] = exp_col_name
        predict_df["reason_idx"] = exp_idx
        predict_df["value"] = max_value

        return predict_df

    def diting_train(self, X, y):
        svm = sklearn.svm.SVC(kernel=self.kernel, probability=True)
        svm.fit(X, y)
        # use Kernel SHAP to explain test set predictions
        x_kmean = shap.kmeans(X, self.n_sample)
        explainer = shap.KernelExplainer(svm.predict_proba, x_kmean, link="logit")
        return explainer

    def diting_test(self, X, explainer):
        shap_values = explainer.shap_values(X, nsamples="auto")
        return shap_values

    def diting_pre(self, df_withlabel):
        df = df_withlabel.drop(['trace_id', 'device_id', 'cluster_id', 'span_name', 'label'], axis=1)
        X = df.values
        y = df_withlabel['label'].values
        columns_name = df.columns.tolist()

        return X, y, columns_name