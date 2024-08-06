
import numpy as np
import ConstructHW
from graph import Graph


class SPLHyper_Model(object):

    def __init__(self, name='SPLHRNMTF'):
        super().__init__()
        self.name = name

    def SPLHRNMTF(self,X,SM,SD,r,alpha,beta,lamda,tol,max_iter):

        m = X.shape[0]
        d = X.shape[1]


        np.random.seed(0)
        U = np.mat(np.random.rand(m, r))
        S = np.mat(np.random.rand(r, r))
        V = np.mat(np.random.rand(d, r))

        '''Hypergraph Learning'''
        graph_SM = np.mat(Graph(np.array(SM), 4))
        spar_SM = np.multiply(SM,graph_SM)

        graph_SD = np.mat(Graph(np.array(SD), 4))
        spar_SD = np.multiply(SD, graph_SD)

        m_embeding = np.mat(np.hstack([X, spar_SM]))
        d_embeding = np.mat(np.hstack([X.T, spar_SD]))
        spar_Dv_m,spar_S_m = ConstructHW.constructHW(m_embeding)
        spar_Dv_d,spar_S_d = ConstructHW.constructHW(d_embeding)


        sample_num = X.shape[1]
        w = np.mat(np.ones(sample_num))
        eps = np.finfo(float).eps


        for niter in range(1,4):

            if niter != 3:
                l = np.mat(np.linalg.norm((X - U * S * V.T), axis=0))
                ls = np.mat(np.sort(l))
                ls_index = np.floor(sample_num * (50 + (niter - 1) * 50) * 0.01).astype(int) - 1
                aeta = ls[:, ls_index]


                zeta = aeta / 2
                for j in range(sample_num):

                    if l[:, j] <= (zeta * aeta) / (zeta + aeta):
                        w[:, j] = 1
                    elif l[:, j] >= aeta:
                        w[:, j] = 0
                    else:
                        w[:, j] = zeta * (1 / l[:, j] - 1 / aeta)
            else:
                w = np.mat(np.ones(sample_num))

            for j in range(max_iter):
                output_X_old = U * S * V.T

                max_value = np.maximum(2 * np.linalg.norm((X - U * S * V.T), 2, axis=0), eps)
                d = w / max_value
                D = np.mat(np.diag(d.tolist()[0]))


                '''Updating the matrix U'''
                temp_U_m = X * D * V * S.T + alpha * spar_S_m * U
                temp_U_d = U * U.T * X * D * V * S.T + alpha * U * U.T * spar_S_m * U
                temp_U = np.mat(np.sqrt((temp_U_m / (temp_U_d + eps))))
                U = np.mat(np.multiply(U, temp_U))

                '''Updating the matrix S'''
                temp_S_m = U.T * X * D * V
                temp_S_d = U.T * U * S * V.T * D * V + lamda * S
                temp_S = np.mat(np.sqrt((temp_S_m / (temp_S_d + eps))))
                S = np.mat(np.multiply(S, temp_S))

                '''Updating the matrix V'''
                temp_V_m = D * X.T * U * S + beta * spar_S_d * V
                temp_V_d = D * V * V.T * X.T * U * S + beta * V * V.T * spar_S_d * V
                temp_V = np.mat(np.sqrt((temp_V_m / (temp_V_d + eps))))
                V = np.mat(np.multiply(V, temp_V))

                output_X = U * S * V.T

                err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
                if err < tol:
                    break

        predict_X = np.array(U * S * V.T)

        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)