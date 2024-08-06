
import pandas as pd
import numpy as np
import os.path as osp
from math import e

class GetData(object):
    def __init__(self,root,miRNA_num = 853,dis_num = 591):
        super().__init__()
        self.root = osp.join(root, 'MDAv3.2_1')
        self.miRNA_num = miRNA_num
        self.dis_num = dis_num
        self.dis_sem_sim, self.mi_fun_sim, self.mi_seq_sim, self.mi_dis_mat = self.__get_data__()

    def __get_data__(self):

        mi_dis_data = pd.read_csv(self.root + '/mi_dis_ass_3.2_1.csv',index_col=0)
        mi_fun_sim = pd.read_csv(self.root + '/mi_fun_sim_3.2_1.csv',index_col=0)
        mi_seq_sim = pd.read_csv(self.root + '/mi_seq_sim_3.2_1.csv', index_col=0)
        dis_sem_sim = pd.read_csv(self.root + '/dis_sem_sim_3.2_1.csv',index_col=0)

        return np.array(dis_sem_sim), np.array(mi_fun_sim), np.array(mi_seq_sim), np.array(mi_dis_data)


    def get_mi_gaussian_sim(self, mir_dis_mat):
        GM = np.zeros((self.miRNA_num, self.miRNA_num))
        rm = self.miRNA_num * 1. / sum(sum(mir_dis_mat * mir_dis_mat))
        for i in range(self.miRNA_num):
            for j in range(self.miRNA_num):
                GM[i][j] = e ** (-rm * (np.dot(mir_dis_mat[i, :] - mir_dis_mat[j, :], mir_dis_mat[i, :] - mir_dis_mat[j, :])))
        return GM

    def get_dis_gaussian_sim(self, mir_dis_mat):
        GD = np.zeros((self.dis_num, self.dis_num))
        T = mir_dis_mat.transpose()
        rd = self.dis_num * 1. / sum(sum(T * T))
        for i in range(self.dis_num):
            for j in range(self.dis_num):
                GD[i][j] = e ** (-rd * (np.dot(T[i,:] - T[j,:], T[i,:] - T[j,:])))
        return GD


