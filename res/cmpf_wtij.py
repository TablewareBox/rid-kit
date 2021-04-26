#!/usr/bin/env python3

import numpy as np
import pandas as pd
import subprocess as sp
from optparse import OptionParser
import os
# from lib.sits_anal import read_sits_dat, compute_log_reweight

def read_sits_dat(filename="sits_enerd.dat"):
    df_sits = pd.read_csv(filename, header=None, delim_whitespace=True)
    if df_sits.shape[1] == 7:
        df_sits.columns = ["E_pp", "E_pw", "E_ww",
                           "E_enh", "E_eff", "reweight", "factor"]
    elif df_sits.shape[1] == 8:
        df_sits.columns = ["step", "E_pp", "E_pw", "E_ww",
                           "E_enh", "E_eff", "reweight", "factor"]

    return df_sits

def compute_log_reweight(E_enh, beta_0, log_nk_i, beta_k_i, log_nk_j=None, beta_k_j=None):
    beta_0k_i = beta_0 - beta_k_i
    log_gf_i = beta_0k_i * E_enh.reshape(-1, 1) + log_nk_i
    log_gf_i -= log_gf_i.mean()
    gfsum_i = np.exp(log_gf_i).sum(axis=1)

    if log_nk_j is not None:
        beta_0k_j = beta_0 - beta_k_j
        log_gf_j = beta_0k_j * E_enh.reshape(-1, 1) + log_nk_j
        log_gf_j -= log_gf_j.mean()
        gfsum_j = np.exp(log_gf_j).sum(axis=1)
        return gfsum_j / gfsum_i
    else:
        return 1. / gfsum_i

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-d', '--jobdir', dest='jobdir', help='Job Directory', default='.')
    parser.add_option('-i', '--idir', dest='idir', help='sits iteration i', default="iter.000001")
    parser.add_option('-j', '--jdir', dest='jdir', help='sits iteration j', default=None)
    P = parser.parse_args()[0]
    print(P)

    tail = 0.90
    cv_dih_dim = None
    data = np.loadtxt("plm.res.out")
    data = data[:, 1:]

    mk_kappa_cmd = "grep KAPPA plumed.res.dat | awk '{print $4}' | cut -d '=' -f 2 > kappa.out"
    sp.check_call(mk_kappa_cmd, shell=True)

    kk = np.loadtxt('kappa.out')
    cc = np.loadtxt('centers.out')
    E = np.array(read_sits_dat("sits_enerd.dat")['E_enh'])
    log_nk_i = np.loadtxt(os.path.join(P.jobdir, P.idir, "log_nk.dat"))
    beta_k_i = np.loadtxt(os.path.join(P.jobdir, P.idir, "beta_k.dat"))
    beta_0 = float(np.loadtxt(os.path.join(P.jobdir, "beta_0.dat")))
    if P.jdir is not None:
        log_nk_j = np.loadtxt(os.path.join(P.jobdir, P.jdir, "log_nk.dat"))
        beta_k_j = np.loadtxt(os.path.join(P.jobdir, P.jdir, "beta_k.dat"))
    else:
        log_nk_j = None
        beta_k_j = None
    weights = compute_log_reweight(E, beta_0, log_nk_i, beta_k_i, log_nk_j, beta_k_j)
    # weights = np.array(read_sits_dat("sits_enerd.dat")['reweight'])

    nframes = data.shape[0]
    ndih_values = data.shape[1]
    if cv_dih_dim is not None:
        ndih_values = cv_dih_dim

    for ii in range(1, nframes):
        for jj in range(ndih_values):
            if data[ii, jj] - data[0, jj] >= np.pi:
                data[ii, jj] -= np.pi * 2.
            elif data[ii, jj] - data[0, jj] < -np.pi:
                data[ii, jj] += np.pi * 2.

    start_f = int(nframes * (1 - tail))
    avgins = np.average(data[start_f:, :], axis=0)
    avgins_wt = np.average(
        data[start_f:, :] * weights[start_f:, None], axis=0) / np.mean(weights[start_f:])

    diff = np.zeros(avgins.shape)
    diff_wt = np.zeros(avgins_wt.shape)
    for ii in range(len(avgins)):
        diff[ii] = avgins[ii] - cc[ii]
        diff_wt[ii] = avgins_wt[ii] - cc[ii]
        if (ii < ndih_values):
            if diff[ii] >= np.pi:
                diff[ii] -= np.pi * 2.
            elif diff[ii] < -np.pi:
                diff[ii] += np.pi * 2.
            if diff_wt[ii] >= np.pi:
                diff_wt[ii] -= np.pi * 2.
            elif diff_wt[ii] < -np.pi:
                diff_wt[ii] += np.pi * 2.

    ff = np.multiply(kk, diff)
    ff_wt = np.multiply(kk, diff_wt)
    if P.jdir is None:
        np.savetxt('force_000.out', np.reshape(ff_wt, [1, -1]), fmt='%.10e')
        np.savetxt('force.out', np.reshape(ff, [1, -1]), fmt='%.10e')
        np.savetxt('force_001.out', np.reshape(ff, [1, -1]), fmt='%.10e')
    else:
        np.savetxt('force.out', np.reshape(ff_wt, [1, -1]), fmt='%.10e')
        np.savetxt('force_%03d.out' % (int(P.idir[-3:])+1), np.reshape(ff_wt, [1, -1]), fmt='%.10e')