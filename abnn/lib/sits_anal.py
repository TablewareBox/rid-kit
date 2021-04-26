#!/usr/bin/env python3

import numpy as np
import pandas as pd
import subprocess as sp


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
    log_gf_i = beta_0k_i * E_enh + log_nk_i
    log_gf_i -= log_gf_i.mean()
    gfsum_i = np.exp(log_gf_i).sum(axis=1)

    if log_nk_j is not None:
        beta_0k_j = beta_0 - beta_k_j
        log_gf_j = beta_0k_j * E_enh + log_nk_j
        log_gf_j -= log_gf_j.mean()
        gfsum_j = np.exp(log_gf_j).sum(axis=1)
        return gfsum_j / gfsum_i
    else:
        return 1. / gfsum_i