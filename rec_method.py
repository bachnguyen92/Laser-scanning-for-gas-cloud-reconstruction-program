import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import Ridge
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gmean
from scipy.sparse import diags
import math
from sklearn.metrics import mean_squared_error


# Root Mean Squared Deviation (NRMSD)
def nrmsd(x_orig, x_rec):
    numerator = np.sum((x_orig - x_rec) ** 2)
    denominator = np.sum((x_orig - np.mean(x_orig)) ** 2)
    return np.sqrt(numerator / denominator)


# Normalized Absolute Average Deviation (NAAAD)
def naaad(x_orig, x_rec):
    numerator = np.sum(np.abs(x_orig - x_rec))
    denominator = np.sum(x_orig)
    return numerator / denominator


# Relative Euclidean Error (ε)
def epsilon(x_orig, x_rec):
    numerator = np.sum(np.abs(x_orig - np.mean(x_orig)) * np.abs(x_rec - np.mean(x_rec)))
    denominator = np.sqrt(np.sum((x_orig - np.mean(x_orig)) ** 2) * np.sum((x_rec - np.mean(x_rec)) ** 2))
    # print(denominator)
    return numerator / denominator

def calculate_relative_change(x_t, x_t_next, norm_type='L2'):
    if norm_type == 'L2':
        # Chuẩn Euclid
        numerator = np.linalg.norm(x_t_next - x_t)
        denominator = np.linalg.norm(x_t)
    elif norm_type == 'L1':
        # Chuẩn Tuyệt Đối
        numerator = np.sum(np.abs(x_t_next - x_t))
        denominator = np.sum(np.abs(x_t))
    elif norm_type == 'Infinity':
        # Chuẩn Tối Đại
        numerator = np.max(np.abs(x_t_next - x_t))
        denominator = np.max(np.abs(x_t))
    else:
        raise ValueError("Unsupported norm type. Use 'L2', 'L1', or 'Infinity'.")

    return numerator / denominator

def SIRT(L_matrix, Y_matrix, V_matrix, w=10):
    # Khởi tạo x
    M, N = L_matrix.shape
    X_matrix = np.zeros(N) + 1e-10
    X_matrix_pre = np.copy(X_matrix)

    # Tạo ma trận C và R
    C = diags(1.0 / (L_matrix.sum(axis=0) + 1e-10))
    R = diags(1.0 / (L_matrix.sum(axis=1) + 1e-10))

    # Tính toán ma trận CATR
    CATR = C @ L_matrix.T @ R

    num_iterations = 0
    while 1:
        num_iterations = num_iterations + 1

        X_matrix = X_matrix + CATR @ (Y_matrix - L_matrix @ X_matrix)

        if calculate_relative_change(X_matrix_pre, X_matrix, norm_type='L2') <= 1e-6:
            break
        else:
            X_matrix_pre = np.copy(X_matrix)

        if num_iterations > 200000:
            break

    return X_matrix, None


def VMR_SART(L_matrix, Y_matrix, V_matrix, w=10):
    M, N = L_matrix.shape

    X_matrix = np.zeros(N) + 1e-10

    X_matrix_pre = np.copy(X_matrix)
    # Tính trước tổng các phần tử trong các hàng và cột
    row_sums = np.sum(L_matrix, axis=1) + 1e-10
    col_sums = np.sum(L_matrix, axis=0) + 1e-10
    num_iterations = 0
    # for k in range(iterations):
    V_matrix = V_matrix.flatten()
    while 1:
        num_iterations = num_iterations + 1
        # Tính residual cho tất cả các hàng cùng một lúc
        residuals = Y_matrix - np.dot(L_matrix, X_matrix)

        # Normalize residuals theo từng hàng
        normalized_residuals = residuals / (row_sums)

        # Tính sum_term cho tất cả các cột cùng một lúc
        sum_term = np.dot(L_matrix.T, normalized_residuals)

        relaxation_factor1 = (V_matrix * X_matrix) / (V_matrix @ X_matrix)  # sử dụng V_matrix

        # w = 10

        X_matrix += ((w * relaxation_factor1) / col_sums) * sum_term

        if calculate_relative_change(X_matrix_pre, X_matrix, norm_type='L2') <= 1e-6:
            break
        else:
            X_matrix_pre = np.copy(X_matrix)

        if num_iterations > 200000:
            break
    print(f"V_SART:{num_iterations}")
    return X_matrix, num_iterations


def Tikhonov(L_matrix, Y_matrix, V_matrix, w=10):
    alphas = np.logspace(-2, 2, 200)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(L_matrix, Y_matrix)
    optimal_alpha = ridge_cv.alpha_

    ridge_model = Ridge(alpha=optimal_alpha)
    ridge_model.fit(L_matrix, Y_matrix)
    X_matrix = ridge_model.coef_

    return X_matrix, None


def SART(L_matrix, Y_matrix, V_matrix, w=10):
    M, N = L_matrix.shape
    X_matrix = np.zeros(N) + 1e-10
    X_matrix_pre = np.copy(X_matrix)

    # Tính trước tổng các phần tử trong các hàng và cột
    row_sums = np.sum(L_matrix, axis=1) + 1e-10
    col_sums = np.sum(L_matrix, axis=0) + 1e-10

    num_iterations = 0
    while 1:
        num_iterations = num_iterations + 1

        # Tính residual cho tất cả các hàng cùng một lúc
        residuals = Y_matrix - np.dot(L_matrix, X_matrix)
        # print("residuals", residuals)

        # Normalize residuals theo từng hàng
        normalized_residuals = residuals / row_sums

        # Tính sum_term cho tất cả các cột cùng một lúc
        sum_term = np.dot(L_matrix.T, normalized_residuals)

        # Update X_matrix cho tất cả các cột cùng một lúc
        X_matrix = X_matrix + (1 / col_sums) * sum_term

        if calculate_relative_change(X_matrix_pre, X_matrix, norm_type='L2') <= 1e-6:
            break
        else:
            X_matrix_pre = np.copy(X_matrix)

        if num_iterations > 200000:
            break

    return X_matrix, num_iterations


def MLEM(L_matrix, Y_matrix, V_matrix, w=10):
    M, N = L_matrix.shape

    X_matrix = np.zeros(N) + 1e-10

    X_matrix_pre = np.copy(X_matrix)

    sum_Aij = np.sum(L_matrix, axis=0) + 1e-10
    num_iterations = 0
    # V_matrix = V_matrix.flatten()
    while 1:
        num_iterations = num_iterations + 1

        sum_Aij_sj = np.dot(L_matrix, X_matrix) + 1e-10

        term1 = X_matrix / (sum_Aij)

        term2 = L_matrix.T @ (Y_matrix / sum_Aij_sj)

        # Update X_matrix cho tất cả các cột cùng một lúc
        X_matrix = term1 * term2

        if calculate_relative_change(X_matrix_pre, X_matrix, norm_type='L2') <= 1e-6:
            break
        else:
            X_matrix_pre = np.copy(X_matrix)

        if num_iterations > 200000:
            break

    print(f"MLEM:{num_iterations}")
    return X_matrix, num_iterations


def LSQR(L_matrix, Y_matrix, V_matrix, w=10):
    result = lsqr(L_matrix, Y_matrix)
    X_matrix = result[0]
    num_iterations = result[2]
    return X_matrix, num_iterations


def NNLS(L_matrix, Y_matrix, V_matrix, w=10):
    result = nnls(L_matrix, Y_matrix, atol=1e-6)
    # result = lsqr(L_matrix, Y_matrix)
    X_matrix = result[0]
    # X_matrix = np.maximum(X_matrix, 0)

    return X_matrix, None