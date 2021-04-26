#!/usr/bin/env python3

import os
from os.path import join
import re
import shutil
import json
import argparse
import numpy as np
import subprocess as sp
import glob
import logging
import time
# global consts
from lib.modeling import cv_dim
from lib.modeling import enhc_name
from lib.modeling import enhc_out_conf
from lib.modeling import enhc_out_angle
from lib.modeling import train_name
from lib.modeling import mol_name
from lib.modeling import mol_files
# utils
from lib.modeling import make_iter_name
from lib.modeling import make_walker_name
from lib.modeling import record_iter
from lib.modeling import log_iter
from lib.modeling import log_task
from lib.modeling import replace
from lib.modeling import make_grompp_enhc, make_grompp_sits
from lib.modeling import copy_file_list
from lib.modeling import create_path
from lib.modeling import cmd_append_log
from lib.modeling import clean_files
# tasks
from lib.modeling import make_res, run_res, post_res, clean_res
from lib.modeling import make_train, run_train, clean_train
from lib.modeling import make_train_eff, train_ori
# machine
import lib.MachineLocal as MachineLocal
import lib.MachineSlurm as MachineSlurm
from lib.machine_exec import exec_hosts
from lib.machine_exec import exec_hosts_batch
from lib.batch_exec import exec_batch
from lib.batch_exec import exec_batch_group

exec_machine = MachineLocal
max_tasks = 1000000

enhc_files = ["plumed.dat", "plumed.bf.dat", "test.std.py", ]
enhc_plm = "plumed.dat"
enhc_bf_plm = "plumed.bf.dat"
enhc_out_plm = "plm.out"


def run_iter(json_file, init_model):
    base_dir = os.getcwd()
    prev_model = init_model
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    sits_param = jdata.get("sits_settings", None)
    numb_iter = jdata["numb_iter"]
    niter_per_sits = sits_param.get("niter_per_sits", 100000000)
    numb_task = 8
    record = "record.train"
    record_sits = "record.sits"
    cleanup = jdata["cleanup"]

    iter_rec = [0, -1]
    sits_iter_rec = [0, -1]
    if os.path.isfile(record):
        with open(record) as frec:
            for line in frec:
                iter_rec = [int(x) for x in line.split()]
        logging.info("continue from iter %03d task %02d" %
                     (iter_rec[0], iter_rec[1]))
    if os.path.isfile(record_sits):
        with open(record_sits) as frec:
            for line in frec:
                sits_iter_rec = [int(x) for x in line.split()]
        logging.info("continue from iter %03d task %02d" %
                     (sits_iter_rec[0], sits_iter_rec[1]))

    global exec_machine

    bPost_train = jdata.get("post_train")

    if sits_iter_rec == [0, -1]:
        create_path("sits")
    data_name = "data"
    for ii in range(iter_rec[0], numb_iter):
        if ii > 0:
            prev_model = glob.glob(make_iter_name(ii-1) + "/" + train_name + "/*pb")
        train_ori(iter_index=ii, json_file=json_file)

        record_iter(record, ii, 0)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("JSON", type=str,
                        help="The json parameter")
    parser.add_argument("-m", "--models", default=[], nargs='*', type=str,
                        help="The init guessed model")
    parser.add_argument("--machine", type=str,
                        help="The machine settings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # logging.basicConfig (filename="compute_string.log", filemode="a", level=logging.INFO, format='%(asctime)s %(message)s')

    machine_type = "local"
    gmxrc = None
    vcores = None
    if args.machine != None:
        fp = open(args.machine, 'r')
        jdata = json.load(fp)
        machine_type = jdata["machine_type"]
        gmxrc = jdata["gmxrc"]
        vcores = jdata["virtual_cores"]

    global exec_machine
    if machine_type == "local":
        exec_machine = MachineLocal
    elif machine_type == "slurm":
        exec_machine = MachineSlurm

    if vcores != None:
        exec_machine.has_virtual_cores(vcores)
    if gmxrc != None:
        exec_machine.add_source_file(gmxrc)

    logging.info("start running")
    run_iter(args.JSON, args.models)
    logging.info("finished!")


if __name__ == '__main__':
    _main()
