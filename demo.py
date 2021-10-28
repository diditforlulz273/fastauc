import timeit
import numpy as np
from fastauc.fast_auc import fast_auc, CppAuc
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 0, 1, 0, 0, 1] * 100, dtype=np.bool8)
    y_prob = np.array([0.1, 0.81, 0.76, 0.1, 0.31, 0.32, 0.34, 0.9] * 100, dtype=np.float32)
    cpp_auc = CppAuc()
    print(f'sklearn AUC = {roc_auc_score(y_true, y_prob)}') 
    print(f'python fast AUC = {fast_auc(y_true, y_prob)}')
    print(f'C++ fast AUC = {cpp_auc.roc_auc_score(y_true, y_prob)}')

    print(f"sklearn AUC 10000 calls time = {timeit.timeit('roc_auc_score(y_true, y_prob)', number=10000, globals=globals())}")
    print(f"python fast AUC 10000 calls time = {timeit.timeit('fast_auc(y_true, y_prob)', number=10000, globals=globals())}")
    print(f"C++ fast AUC 10000 calls time = {timeit.timeit('cpp_auc.roc_auc_score(y_true, y_prob)', number=10000, globals=globals())}")   

