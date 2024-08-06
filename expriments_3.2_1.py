

import numpy as np
import sim_fusion
from data import MDAv3_1
from methods import model
from WKNN import WKNN_method


class Experiments(object):
    def __init__(self,mir_dis_data, model_name='SPLHRNMTF', **kwargs):
        super().__init__()
        self.mir_dis_data = mir_dis_data
        self.model = model.SPLHyper_Model(model_name)
        self.parameters = kwargs


    def CV(self):
        k_folds = 5
        association_matrix = self.mir_dis_data.mi_dis_mat
        index_matrix = np.array(np.where(association_matrix > 0))

        pair_num = index_matrix.shape[1]
        sample_num_per_fold = int(pair_num / k_folds)
        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics = np.zeros((1, 7))
        for k in range(k_folds):
            print('The {}th cross validation'.format(k+1))
            train_matrix = np.array(self.mir_dis_data.mi_dis_mat, copy=True)

            if k != k_folds - 1:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_matrix[test_index] = 0

            mi_fun_sim_arr = self.mir_dis_data.mi_fun_sim
            mi_seq_sim_arr = self.mir_dis_data.mi_seq_sim
            mi_gau_sim_arr = self.mir_dis_data.get_mi_gaussian_sim(train_matrix)


            dis_sem_sim_arr = self.mir_dis_data.dis_sem_sim
            dis_gau_sim_arr = self.mir_dis_data.get_dis_gaussian_sim(train_matrix)


            k1 = int(mi_fun_sim_arr.shape[0] / 10)
            k2 = int(dis_sem_sim_arr.shape[0] / 10)

            '''miRNA'''
            m1 = sim_fusion.new_normalization(mi_fun_sim_arr)
            m2 = sim_fusion.new_normalization(mi_seq_sim_arr)
            m3 = sim_fusion.new_normalization(mi_gau_sim_arr)
            Sm_1 = sim_fusion.KNN_kernel(mi_fun_sim_arr, k1)
            Sm_2 = sim_fusion.KNN_kernel(mi_seq_sim_arr, k1)
            Sm_3 = sim_fusion.KNN_kernel(mi_gau_sim_arr, k1)

            Pm = sim_fusion.miRNA_updating(Sm_1, Sm_2, Sm_3, m1, m2, m3)
            Pm_final = (Pm + Pm.T) / 2

            '''disease'''
            d1 = sim_fusion.new_normalization(dis_sem_sim_arr)
            d2 = sim_fusion.new_normalization(dis_gau_sim_arr)
            Sd_1 = sim_fusion.KNN_kernel(dis_sem_sim_arr, k2)
            Sd_2 = sim_fusion.KNN_kernel(dis_gau_sim_arr, k2)

            Pd = sim_fusion.disease_updating(Sd_1, Sd_2, d1, d2)
            Pd_final = (Pd + Pd.T) / 2

            train_matrix_data = np.mat(train_matrix)
            Pm_final_mat = np.mat(Pm_final)
            Pd_final_mat = np.mat(Pd_final)

            '''improved association'''
            new_train_matrix = WKNN_method(train_matrix_data, Pm_final_mat, Pd_final_mat, 10, 0.9)
            new_train_matrix_data = np.mat(new_train_matrix)


            predict_mat = self.model()(new_train_matrix_data,Pm_final_mat,Pd_final_mat,
                                       r=self.parameters['r'], alpha=self.parameters['alpha'],beta=self.parameters['beta'],
                                       lamda=self.parameters['lamda'],tol=5e-4, max_iter=2000)

            for i in range(10):
                metrics = metrics + self.cv_mat_model_evaluate(self.mir_dis_data.mi_dis_mat,
                                                                  predict_mat,test_index, i)

        result = np.around(metrics / 50, decimals=8)

        return result


    def cv_mat_model_evaluate(self, association_mat, predict_mat, test_index, seed):
        test_po_num = np.array(test_index).shape[1]
        test_index_0 = np.array(np.where(association_mat == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index_0.T)

        test_ne_index = tuple(test_index_0[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_mat[test_ne_index].flatten()), np.mat(association_mat[test_index].flatten())))

        predict_score = np.column_stack(
            (np.mat(predict_mat[test_ne_index].flatten()), np.mat(predict_mat[test_index].flatten())))

        # real_score and predict_score are array
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 2000) / np.array([2000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]


if __name__ == '__main__':
    root = './HMDD_data'

    mir_dis_data = MDAv3_1.GetData(root)
    experiment = Experiments(mir_dis_data, model_name='SPLHRNMTF', r = 46, alpha = 0.2, beta = 0.002, lamda = 0.0001,
                             tol=5e-4, max_iter=2000)
    print(experiment.CV())









