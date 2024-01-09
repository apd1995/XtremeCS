#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 6 18:16:37 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame
import time
import json
import gc
from EMS.manager import do_on_cluster, get_gbq_credentials, do_test_experiment, read_json, unroll_experiment
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import dask
import logging

logging.basicConfig(level=logging.INFO)
log_gbq = logging.getLogger('pandas_gbq')
log_gbq.setLevel(logging.DEBUG)
log_gbq.addHandler(logging.StreamHandler())


logging.basicConfig(level=logging.INFO)


def seed(nonzero_rows_exponent: float, nonzero_rows_coef: float, signal_dim: int, num_measurements: int, mc: int, err_tol: float, sparsity_tol: float) -> int:
    return round(1 + round(nonzero_rows_exponent * 1000) + round(nonzero_rows_coef * 1000) + round(signal_dim * 1000) + round(num_measurements * 1000) + round(err_tol * 100000) + mc * 100000 + round(sparsity_tol * 1000000))


def _df(c: list, l: list) -> DataFrame:
    d = dict(zip(c, l))
    return DataFrame(data=d, index=[0])


def df_experiment(sparsity: float,
                  delta: float,
                  N: int,
                  B: int,
                  k: int,
                  n: int,
                  mc: int,
                  err_tol: float,
                  rel_err: float,
                  avg_err: float,
                  max_row_err: float,
                  norm_2_1_true: float,
                  norm_2_1_rec: float,
                  norm_2_2_true: float,
                  norm_2_2_rec: float,
                  norm_2_infty_true: float,
                  norm_2_infty_rec: float,
                  sparsity_tol: float,
                  soft_sparsity: float,
                  nonzero_rows_rec: int,
                  tpr: float,
                  tnr: float,
                  time_seconds: float) -> DataFrame:
    c = ['sparsity', 'delta', 'signal_nrow', 'B', 'k', 'num_measurements', 'mc', 'err_tol', 'rel_err', 'avg_err', 'max_row_err',
         'norm_2_1_true', 'norm_2_1_rec', 'norm_2_2_true', 'norm_2_2_rec', 'norm_2_infty_true',
         'norm_2_infty_rec', 'sparsity_tol', 'soft_sparsity', 'nonzero_rows_rec', 'tpr', 'tnr', 'time_seconds']
    d = [sparsity, delta, N, B, k, n, mc, err_tol, rel_err, avg_err, max_row_err,
         norm_2_1_true, norm_2_1_rec, norm_2_2_true, norm_2_2_rec, norm_2_infty_true,
         norm_2_infty_rec, sparsity_tol, soft_sparsity, nonzero_rows_rec, tpr, tnr, time_seconds]
    return _df(c, d)


def gen_iid_normal_mtx(n, N, rng: Generator):
    """
    Generates a single random n by N matrix with iid N(0,1) entries

    Parameters
    ----------
    n : int
        Number of rows of measurement matrix.
    N : int
        Number of rows of signal matrix.

    Returns
    -------
    numpy.ndarray
        n by N matrix.

    """
    
    return rng.normal(0, 1, (n, N))


def cvx_stats(A, x, y, sparsity_tol):
    """
    x is the true signal, A is the measurement matrix and y = Ax.
    Given y and A, solves the convex optimization:
        min. ell_2,1 norm of v
        such that Av = y
    and returns the relative error norm(x - v)/norm(x)

    Parameters
    ----------
    A : numpy.ndarray
        Measurement matrix (must have no. of columns same as no. of rows of x).
    x : numpy.ndarray
        True signal matrix.
    y : numpy.ndarray
        Measured signal y = Ax.

    Returns
    -------
    out_dict: dict
        Various stats of recovered signal matrix.

    """
    start_time = time.time()
    vx = cvx.Variable(x.shape, complex=False)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A@vx == y]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(verbose=False, warm_start = True)
    rel_err = cvx.norm(vx.value-x, 2).value/cvx.norm(x, 2).value,
    avg_err = cvx.norm(vx.value-x, 2).value/np.sqrt(x.shape[0]),
    soft_sparsity = np.mean(np.abs(vx.value) > sparsity_tol)
    nonzero_rows_rec = int(soft_sparsity * x.shape[0])
    zero_indices_true = (x == 0)
    zero_indices_rec = (np.abs(vx.value)<=sparsity_tol)
    tpr = sum(zero_indices_true * zero_indices_rec)/max(1, sum(zero_indices_true))
    nonzero_indices_true = (x != 0)
    nonzero_indices_rec = (np.abs(vx.value) > sparsity_tol)
    tnr = sum(nonzero_indices_true * nonzero_indices_rec)/max(1, sum(nonzero_indices_true))
    end_time = time.time()
    time_seconds = round(end_time - start_time, 2)
    
    out_dict = {'rel_err': rel_err,
                    'avg_err': avg_err,
                    'soft_sparsity': soft_sparsity,
                    'nonzero_rows_rec': nonzero_rows_rec,
                    'tpr': tpr,
                    'tnr': tnr,
                    'time_seconds': time_seconds}
    return out_dict


def run_basis_pursuit(**dict_params) -> DataFrame:
    """
    Generates random instance of ell_2,1 norm minimization.
    Returns success or failure according to recovery of true signal upto rel_err
    relative error.

    Parameters
    ----------
    sparsity : float
        Fraction of non-zero rows in signal matrix.
    delta : float
        Undersampling fraction.
    N : int
        Number of rows in signal matrix.
    B : int
        Number of columns in signal matrix.
    rel_err_tol: float.
        Relative error allowed to declare success in recovering true signal.
    mc: int.
        Monte Carlo run count.
    sparsity_tol: float.
        Sparsity tolerance provided by user to decide when a row of recovered matrix
        will be deemed non-zero.

    Returns
    -------
    success : int
        Success (1) or failure (0).

    """
    nonzero_rows_exponent = dict_params['nonzero_rows_exponent']
    nonzero_rows_coef = dict_params['nonzero_rows_coef']
    N = dict_params['signal_dim']
    k = dict_params['nonzero_rows']
    n = dict_params['num_measurements']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    rng = np.random.default_rng(seed=seed(nonzero_rows_exponent, nonzero_rows_coef, N, n, mc, err_tol, sparsity_tol))

    gc.collect()
    x = np.zeros(N, dtype=float)
    indices = rng.choice(range(N), k, replace=False)
    x[indices] = rng.normal(0, 1, len(indices))
    
    A = gen_iid_normal_mtx(n, N, rng)
    y = A @ x
    dict_observables = cvx_stats(A, x, y, sparsity_tol)
    
    combined_dict = {**dict_params, **dict_observables}

    return DataFrame(combined_dict, index = [0])


def test_experiment() -> dict:
    exp = {'table_name':'BP_sublinear_sparsity',
           'params': [{
               'nonzero_rows_exponent': [0.5],
               'nonzero_rows_coef': [1],
               'signal_dim': [10**4],
               'num_measurements': [x for x in range(100, 5000, 50)],
               'err_tol': [1e-5],
               'sparsity_tol': [1e-4],
               'mc': [x for x in range(20)]
                }]
           }
    return exp


def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    nodes = 1000
    with SLURMCluster(queue='normal,owners,donoho,hns,stat',
                      cores=1, memory='10GiB', processes=1,
                      walltime='24:00:00') as cluster:
        #cluster.scale(jobs=nodes)
        cluster.adapt(minimum = nodes, maximum = nodes, target_duration = "86400s", interval = "3000ms")
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, run_basis_pursuit, client, credentials=get_gbq_credentials())
        cluster.scale(0)


def do_local_experiment():
    exp = test_experiment()
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_basis_pursuit, client, credentials=None)
            # do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787', n_workers=2) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_basis_pursuit, client, credentials=None)
            # do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def do_test_exp():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_test_experiment(exp, run_basis_pursuit, client, credentials=get_gbq_credentials())


def do_test():
    exp = test_experiment()
    print(exp)
    pass
    df = run_basis_pursuit(dict_params = exp)
    df.to_csv("temp.csv")


def count_params(json_file: str):
    exp = read_json(json_file)
    params = unroll_experiment(exp)
    logging.info(f'Count of instances: {len(params)}.')

if __name__ == '__main__':
    # do_test_exp()
    # read_and_do_local_experiment('exp_dicts/BP_sublinear_sparsity.json')
    gc.collect()
    do_sherlock_experiment('exp_dicts/BP_sublinear_sparsity.json')

