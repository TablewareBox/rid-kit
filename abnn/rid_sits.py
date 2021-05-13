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
from lib.modeling import make_train_eff
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


def make_enhc(iter_index,
              json_file,
              graph_files,
              sits_iter=False):
    graph_files.sort()
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    bPosre = jdata.get("gmx_posre", False)
    sits_param = jdata.get("sits_settings", None)
    if sits_param is not None:
        sits_param["nst-sits-enerd-out"] = jdata["bias_frame_freq"]
    
    numb_walkers = jdata["numb_walkers"] if not sits_iter else 1
    template_dir = jdata["template_dir"]
    enhc_trust_lvl_1 = jdata["bias_trust_lvl_1"]
    enhc_trust_lvl_2 = jdata["bias_trust_lvl_2"]

    nsteps = jdata["bias_nsteps"]
    if sits_param is not None:
        if sits_iter:
            nsteps = sits_param["sits_nsteps"]
    frame_freq = jdata["bias_frame_freq"]
    num_of_cluster_threshhold = jdata["num_of_cluster_threshhold"]

    iter_name = make_iter_name(iter_index)
    if sits_param is not None:
        if sits_iter:
            iter_name = join("sits", make_iter_name(iter_index))
            work_path = iter_name + "/"
    work_path = iter_name + "/" + enhc_name + "/"
    mol_path = template_dir + "/" + mol_name + "/"
    enhc_path = template_dir + "/" + enhc_name + "/"
    conf_list = glob.glob(mol_path + "conf*gro")
    conf_list.sort()
    assert (len(conf_list) >=
            numb_walkers), "not enough conf files in mol dir %s" % mol_path

    create_path(work_path)

    for walker_idx in range(numb_walkers):
        walker_path = work_path + make_walker_name(walker_idx) + "/"
        create_path(walker_path)
        if sits_param is not None:
            if os.path.exists(join("sits", "log_nk.dat")):
                try:
                    shutil.copyfile(join("sits", "log_nk.dat"), join(walker_path, "log_nk.dat"))
                except:
                    pass
            if os.path.exists(join("sits", "log_norm.dat")):
                try:
                    shutil.copyfile(join("sits", "log_norm.dat"), join(walker_path, "log_norm.dat"))
                except:
                    pass
        # copy md ifles
        for ii in mol_files:
            if os.path.exists(walker_path + ii):
                os.remove(walker_path + ii)
            try:
                shutil.copy(mol_path + ii, walker_path)
            except:
                pass
        # copy conf file
        conf_file = conf_list[walker_idx]
        if os.path.exists(walker_path + "conf.gro"):
            os.remove(walker_path + "conf.gro")
        try:
            shutil.copy(conf_file, walker_path + "conf.gro")
        except:
            pass

        # if have prev confout.gro, use as init conf
        if (iter_index > 0):
            prev_enhc_path = make_iter_name(
                iter_index-1) + "/" + enhc_name + "/" + make_walker_name(walker_idx) + "/"
            prev_enhc_path = os.path.abspath(prev_enhc_path) + "/"
            if os.path.isfile(prev_enhc_path + "confout.gro"):
                os.remove(walker_path + "conf.gro")
                os.symlink(prev_enhc_path + "confout.gro",
                           walker_path + "conf.gro")
            else:
                raise RuntimeError(
                    "cannot find prev output conf file  " + prev_enhc_path + 'confout.gro')
            log_task("use conf of iter " + make_iter_name(iter_index -
                     1) + " walker " + make_walker_name(walker_idx))
###########################################
            num_of_cluster = np.loadtxt(prev_enhc_path+'num_of_cluster.dat')
            pre_trust_lvl1 = np.loadtxt(prev_enhc_path+'trust_lvl1.dat')
            if num_of_cluster < num_of_cluster_threshhold:
                enhc_trust_lvl_1 = pre_trust_lvl1 * 1.5
                enhc_trust_lvl_2 = enhc_trust_lvl_1+1
            else:
                enhc_trust_lvl_1 = jdata["bias_trust_lvl_1"]
                enhc_trust_lvl_2 = enhc_trust_lvl_1+1
            if enhc_trust_lvl_1 > jdata["bias_trust_lvl_1"]*8:
                enhc_trust_lvl_1 = jdata["bias_trust_lvl_1"]
                enhc_trust_lvl_2 = enhc_trust_lvl_1+1

        np.savetxt(walker_path+'trust_lvl1.dat',
                   [enhc_trust_lvl_1], fmt='%.6f')
        # copy enhc file
        for ii in enhc_files:
            if os.path.exists(walker_path + ii):
                os.remove(walker_path + ii)
            try:
                shutil.copy(enhc_path + ii, walker_path)
            except:
                pass
        # copy graph files
        for ii in graph_files:
            file_name = os.path.basename(ii)
            abs_path = os.path.abspath(ii)
            if os.path.exists(walker_path + file_name):
                os.remove(walker_path + file_name)
            os.symlink(abs_path, walker_path + file_name)
        # config MD
        mol_conf_file = walker_path + "grompp.mdp"
        if bPosre:
            mol_conf_file = walker_path + "grompp_restraint.mdp"
        if sits_param is not None:
            if sits_iter:
                mol_conf_file = walker_path + "grompp_sits_iter.mdp"
            else:
                mol_conf_file = walker_path + "grompp_sits.mdp"
            if bPosre:
                gmx_prep = gmx_prep.replace(".mdp", "_restraint.mdp")
            make_grompp_sits(mol_conf_file, sits_param, sits_iter=sits_iter, iter_index=iter_index)

        make_grompp_enhc(mol_conf_file, nsteps, frame_freq)
        # config plumed
        graph_list = ""
        counter = 0
        for ii in graph_files:
            file_name = os.path.basename(ii)
            if counter == 0:
                graph_list = "%s" % file_name
            else:
                graph_list = "%s,%s" % (graph_list, file_name)
            counter = counter + 1
        plm_conf = walker_path + enhc_plm
        replace(plm_conf, "MODEL=[^ ]* ", ("MODEL=%s " % graph_list))
        replace(plm_conf, "TRUST_LVL_1=[^ ]* ",
                ("TRUST_LVL_1=%f " % enhc_trust_lvl_1))
        replace(plm_conf, "TRUST_LVL_2=[^ ]* ",
                ("TRUST_LVL_2=%f " % enhc_trust_lvl_2))
        replace(plm_conf, "STRIDE=[^ ]* ", ("STRIDE=%d " % frame_freq))
        replace(plm_conf, "FILE=[^ ]* ", ("FILE=%s " % enhc_out_plm))

        plm_bf_conf = walker_path + enhc_bf_plm
        replace(plm_bf_conf, "STRIDE=[^ ]* ", ("STRIDE=%d " % frame_freq))
        replace(plm_bf_conf, "FILE=[^ ]* ", ("FILE=%s " % enhc_out_plm))

        if len(graph_list) == 0:
            log_task("brute force MD without NN acc")
        else:
            log_task("use NN model(s): " + graph_list)
            log_task("set trust l1 and l2: %f %f" %
                     (enhc_trust_lvl_1, enhc_trust_lvl_2))


def run_enhc(iter_index,
             json_file,
             sits_iter=False):
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    cmd_env = jdata.get("cmd_sources", [])
    sits_param = jdata.get("sits_settings", None)
    bPosre = jdata.get("gmx_posre", False)

    iter_name = make_iter_name(iter_index)
    if sits_param is not None:
        if sits_iter:
            iter_name = join("sits", make_iter_name(iter_index))
    work_path = iter_name + "/" + enhc_name + "/"

    gmx_prep = jdata["gmx_prep"]
    if sits_param is not None:
        if sits_iter:
            gmx_prep += " -f grompp_sits_iter.mdp"
        else:
            gmx_prep += " -f grompp_sits.mdp"
        if bPosre:
            gmx_prep = gmx_prep.replace(".mdp", "_restraint.mdp")
    elif bPosre:
        gmx_prep += " -f grompp_restraint.mdp"
    gmx_run = jdata["gmx_run"]
    enhc_thread = jdata["bias_thread"]
    gmx_run = gmx_run + (" -nt %d " % enhc_thread)
    gmx_prep_log = "gmx_grompp.log"
    gmx_run_log = "gmx_mdrun.log"
    # assuming at least one walker
    graph_files = glob.glob(work_path + (make_walker_name(0)) + "/*.pb")
    if len(graph_files) != 0:
        gmx_run = gmx_run + " -plumed " + enhc_plm
    else:
        gmx_run = gmx_run + " -plumed " + enhc_bf_plm
    gmx_prep_cmd = cmd_append_log(gmx_prep, gmx_prep_log, env=cmd_env)
    gmx_run_cmd = cmd_append_log(gmx_run, gmx_run_log, env=cmd_env)
    numb_walkers = jdata["numb_walkers"] if not sits_iter else 1
    batch_jobs = jdata['batch_jobs']
    batch_time_limit = jdata['batch_time_limit']
    batch_modules = jdata['batch_modules']
    batch_sources = jdata['batch_sources']

    all_task = glob.glob(work_path + "/[0-9]*[0-9]")
    all_task.sort()

    global exec_machine
    exec_hosts(MachineLocal, gmx_prep_cmd, 1, all_task, None)
    if batch_jobs:
        exec_batch(gmx_run_cmd, enhc_thread, 1, all_task, task_args=None,
                   time_limit=batch_time_limit, modules=batch_modules, sources=batch_sources)
    else:
        if len(all_task) == 1:
            exec_hosts(MachineLocal, gmx_run_cmd, enhc_thread, all_task, None)
        else:
            exec_hosts_batch(exec_machine, gmx_run_cmd,
                             enhc_thread, all_task, None)


def post_enhc(iter_index,
              json_file):
    iter_name = make_iter_name(iter_index)
    work_path = iter_name + "/" + enhc_name + "/"

    fp = open(json_file, 'r')
    jdata = json.load(fp)
    cmd_env = jdata.get("cmd_sources", [])
    gmx_split = jdata["gmx_split_traj"]
    gmx_split_log = "gmx_split.log"
    gmx_split_cmd = cmd_append_log(gmx_split, gmx_split_log, env=cmd_env)

    all_task = glob.glob(work_path + "/[0-9]*[0-9]")
    all_task.sort()

    cwd = os.getcwd()
    numb_walkers = jdata["numb_walkers"]
    for ii in range(numb_walkers):
        walker_path = work_path + make_walker_name(ii) + "/"
        os.chdir(walker_path)
        if os.path.isdir("confs"):
            try:
                shutil.rmtree("confs")
            except:
                pass
        os.makedirs("confs")
        os.chdir(cwd)

    global exec_machine
    exec_hosts(MachineLocal, gmx_split_cmd, 1, all_task, None)

    for ii in range(numb_walkers):
        walker_path = work_path + make_walker_name(ii) + "/"
        angles = np.loadtxt(walker_path + enhc_out_plm)
        np.savetxt(walker_path + enhc_out_angle, angles[:, 1:], fmt="%.6f")


def clean_enhc(iter_index):
    iter_name = make_iter_name(iter_index)
    work_path = iter_name + "/" + enhc_name + "/"

    all_task = glob.glob(work_path + "/[0-9]*[0-9]")
    all_task.sort()

    cleaned_files = ['state*.cpt', '*log', 'traj.trr',
                     'topol.tpr', 'ener.edr', 'mdout.mdp']
    cwd = os.getcwd()
    for ii in all_task:
        os.chdir(ii)
        clean_files(cleaned_files)
        os.chdir(cwd)


def clean_enhc_confs(iter_index):
    iter_name = make_iter_name(iter_index)
    work_path = iter_name + "/" + enhc_name + "/"

    all_task = glob.glob(work_path + "/[0-9]*[0-9]")
    all_task.sort()

    cwd = os.getcwd()
    for ii in all_task:
        os.chdir(ii)
        sel_idx = np.loadtxt('cls.sel.out', dtype=int)
        pres_files = []
        if sel_idx.size == 0:
            os.chdir(cwd)
            continue
        if sel_idx.size == 1:
            sel_idx = np.array([sel_idx], dtype=int)
        for jj in sel_idx:
            conf_file = os.path.join('confs', 'conf%d.gro' % jj)
            pres_files.append(conf_file)
        all_files = glob.glob(os.path.join('confs', 'conf*gro'))
        for jj in all_files:
            if not (jj in pres_files):
                os.remove(jj)
        os.chdir(cwd)

def make_sits_iter(sits_iter_index, json_file, graph_files):
    make_enhc(sits_iter_index, json_file, graph_files, sits_iter=True)
    if sits_iter_index > 0:
        old_dir = join("sits", make_iter_name(sits_iter_index-1))
        walker_dir = join("sits", make_iter_name(sits_iter_index), enhc_name, make_walker_name(0))
        try:
            shutil.copyfile( join(old_dir, "log_nk.dat"), join(walker_dir, "log_nk.dat") )
            shutil.copyfile( join(old_dir, "log_norm.dat"), join(walker_dir, "log_norm.dat") )
        except:
            pass


def run_sits_iter(sits_iter_index, json_file):
    run_enhc(sits_iter_index, json_file, sits_iter=True)

def post_sits_iter(sits_iter_index, json_file):

    sits_dir = join("sits", make_iter_name(sits_iter_index))
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    sits_param = jdata.get("sits_settings", None)
    tempf = np.linspace(sits_param["sits-t-low"], sits_param["sits-t-high"], sits_param["sits-t-numbers"])

    CONSTANT_kB = 0.008314472
    beta_k = 1.0 / (CONSTANT_kB * tempf)
    beta_0 = 1.0 / (CONSTANT_kB * sits_param["sits-t-ref"])
    if not os.path.exists(join(sits_dir, "beta_k.dat")):
        np.savetxt(join(sits_dir, "beta_k.dat"), beta_k)
    if not os.path.exists(join("sits", "beta_0.dat")):
        np.savetxt(join("sits", "beta_0.dat"), np.array([beta_0]))
    walker_dir = join(sits_dir, enhc_name, make_walker_name(0))

    cmd_save_sits = ""
    cmd_save_sits += "tail -n 1 %s > %s\n" % (join(walker_dir, "sits_nk.dat"), join(sits_dir, "log_nk.dat"))
    cmd_save_sits += "tail -n 1 %s > %s\n" % (join(walker_dir, "sits_norm.dat"), join(sits_dir, "log_norm.dat"))
    cmd_save_sits += "cp %s %s\n" % (join(sits_dir, "log_nk.dat"), join("sits", "log_nk.dat"))
    cmd_save_sits += "cp %s %s\n" % (join(sits_dir, "log_norm.dat"), join("sits", "log_norm.dat"))
    cmd_save_sits += "cp %s %s\n" % (join(sits_dir, "beta_k.dat"), join("sits", "beta_k.dat"))
    sp.check_call(cmd_save_sits, shell=True)


def run_train_eff(sits_iter_index, json_file, exec_machine=MachineLocal):
    run_train(sits_iter_index, json_file, exec_machine=exec_machine, data_name="data%03d" % (sits_iter_index + 1), sits_iter=True)

def post_train_eff(sits_iter_index, json_file):
    # copy trained model in sits_train_path to last rid iter (prev_*)
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    template_dir = jdata["template_dir"]
    numb_model = jdata["numb_model"]
    base_path = os.getcwd() + "/"
    if sits_iter_index > 0:
        sits_iterj_name = join("sits", make_iter_name(sits_iter_index - 1))
        sits_rid_iter = np.array([np.loadtxt(join(sits_iterj_name, "rid_iter_begin.dat")),
                                  np.loadtxt(join(sits_iterj_name, "rid_iter_end.dat"))]).astype(int)
        iter_end = int(sits_rid_iter[1])
        prev_iter_index = iter_end - 1
        prev_iter_name = make_iter_name(prev_iter_index)
        prev_train_path = prev_iter_name + "/" + train_name + "/"
        prev_train_path = os.path.abspath(prev_train_path) + "/"

        sits_iter_name = join("sits", make_iter_name(sits_iter_index))

        data_dir = "data"
        data_name = "data%03d" % sits_iter_index
        train_path = join(sits_iter_name, train_name)

        for ii in range(numb_model):
            work_path = join(train_path, ("%03d" % ii))
            model_files = glob.glob(join(work_path, "model.ckpt.*")) + [join(work_path, "checkpoint")]

            prev_work_path = prev_train_path + ("%03d/" % ii)
            prev_model_files = glob.glob(join(prev_work_path, "model.ckpt.*")) + [join(prev_work_path, "checkpoint")]
            # prev_model_files += [join(prev_work_path, "checkpoint")]
            old_model_path = join(prev_work_path, "old_model")
            create_path(old_model_path)
            os.chdir(old_model_path)
            for mfile in model_files:
                os.symlink(os.path.relpath(mfile), os.path.basename(mfile))
                # shutil.copy (ii, old_model_path)
            os.chdir(base_path)
            for mfile in model_files:
                if os.path.exists(join(prev_work_path, mfile)):
                    os.rename(join(prev_work_path, mfile), join(prev_work_path, mfile) + ".%03d" % sits_iter_index)
                try:
                    shutil.copy(mfile, prev_work_path)
                except:
                    pass

            prev_models = glob.glob(join(prev_train_path, "*.pb"))
            models = glob.glob(join(train_path, "*.pb"))
            for mfile in models:
                model_name = os.path.basename(mfile)
                if os.path.exists(join(prev_train_path, model_name)):
                    os.rename(join(prev_train_path, model_name), join(prev_train_path, model_name) + ".%03d" % sits_iter_index)
                os.symlink(os.path.abspath(mfile), os.path.abspath(join(prev_train_path, model_name)))


def run_iter(json_file, init_model):
    base_dir = os.getcwd()
    prev_model = init_model
    fp = open(json_file, 'r')
    jdata = json.load(fp)
    sits_param = jdata.get("sits_settings", None)
    numb_iter = jdata["numb_iter"]
    niter_per_sits = sits_param.get("niter_per_sits", 100000000)
    numb_task = 8
    record = "record.rid"
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
        if ii % niter_per_sits == 0:
            kk = int(ii / niter_per_sits)
            log_iter("run_sits_iter", kk, 0)
            if not os.path.exists(join("sits", make_iter_name(kk))):
                create_path(join("sits", make_iter_name(kk)))
            if kk > 0:
                open(join("sits", make_iter_name(kk-1), "rid_iter_end.dat"), "w+").write("%d" % ii)
            open(join("sits", make_iter_name(kk), "rid_iter_begin.dat"), "w+").write("%d" % ii)
            for jj in range(sits_iter_rec[1], 6):
                if kk * max_tasks + jj <= sits_iter_rec[0] * max_tasks + sits_iter_rec[1]:
                    continue
                os.chdir(base_dir)
                if jj == 0:
                    make_sits_iter(kk, json_file, prev_model)
                elif jj == 1:
                    run_sits_iter(kk, json_file)
                elif jj == 2:
                    post_sits_iter(kk, json_file)
                elif jj == 3:
                    if kk > 0:
                        make_train_eff(kk, json_file)
                elif jj == 4:
                    if kk > 0:
                        run_train_eff(kk, json_file, exec_machine)
                elif jj == 5:
                    if kk > 0:
                        post_train_eff(kk, json_file)
                record_iter(record_sits, kk, jj)
            data_name = "data%03d" % (kk+1)
        
        for jj in range(numb_task):
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                continue
            os.chdir(base_dir)
            if jj == 0:
                log_iter("make_enhc", ii, jj)
                # logging.info ("use prev model " + str(prev_model))
                make_enhc(ii, json_file, prev_model)
            elif jj == 1:
                log_iter("run_enhc", ii, jj)
                run_enhc(ii, json_file)
            elif jj == 2:
                log_iter("post_enhc", ii, jj)
                post_enhc(ii, json_file)
            elif jj == 3:
                log_iter("make_res", ii, jj)
                cont = make_res(ii, json_file)
                if not cont:
                    log_iter("no more conf needed", ii, jj)
                    return
            elif jj == 4:
                log_iter("run_res", ii, jj)
                run_res(ii, json_file, exec_machine)
            elif jj == 5:
                log_iter("post_res", ii, jj)
                post_res(ii, json_file, data_name=data_name)
            elif jj == 6:
                log_iter("make_train", ii, jj)
                make_train(ii, json_file, data_name=data_name)
            elif jj == 7:
                log_iter("run_train", ii, jj)
                run_train(ii, json_file, exec_machine, data_name=data_name)
                if cleanup:
                    clean_train(ii)
                    clean_enhc(ii)
                    clean_enhc_confs(ii)
                    clean_res(ii)
            else:
                raise RuntimeError("unknow task %d, something wrong" % jj)

            record_iter(record, ii, jj)


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
