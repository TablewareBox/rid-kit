#!/usr/bin/env python3
import os
import mdtraj as md
import numpy as np
import pickle
import pathlib
import re
import shutil
import time
import heapq
import argparse

"""
This file can make rid dir for given molecular.
Please modify 'pdbname'.
Last update date: 2021/2/24
Author: Dongdong Wang, Yanze Wang.
"""

ridkit_dir = "~/cjh/loopOpt_test"


def replace(file_name, pattern, subst):
    """
    Replace string in file_name. from pattern to subst. pattern is written by Regular Expression.
    """
    file_handel = open(file_name, 'r')
    file_string = file_handel.read()
    file_handel.close()
    file_string = (re.sub(pattern, subst, file_string))
    file_handel = open(file_name, 'w')
    file_handel.write(file_string)
    file_handel.close()


def change_his(pdbname):
    """
    This function can change all the HIS residues to HSD residues in pdbfile(pdbname).
    it's used to meet the need of force field file. Some pdbs have different H+ sites on HIS residues, varying from HSD to HSP.
    """
    with open(pdbname, 'r') as pdb:
        ret = pdb.read()
    ret = ret.replace('HIS', 'HSD')
    with open(pdbname, 'w') as pdb:
        pdb.write(ret)


def pbs_submit(title, command, dir_path=os.getcwd(), host='V100_8_32', is_wait=True):
    """
    This is a very naive pbs submittion function. for a given command, this function will generate a .pbs file to be submited.
    The process will not be terminated untill the jobs in PBS are done.
    We recommand dpdisphier package as a better and mature tool.
    """
    if is_wait:
        is_running = True
        with open('pbs_stat_{}.log'.format(title), 'w') as log:
            log.write('Running')
    with open('{}.pbs'.format(title), 'w') as wf:
        wf.write('#!/bin/bash -l\n')
        wf.write('#PBS -N {}\n'.format(title))
        wf.write('#PBS -o {}/log_{}.out\n'.format(dir_path, title))
        wf.write('#PBS -e {}/log_{}.err\n'.format(dir_path, title))
        wf.write('#PBS -l select=1:ncpus=4:ngpus=1\n')
        wf.write('#PBS -l walltime=120:0:0\n')
        wf.write('#PBS -q {}\n'.format(host))
        wf.write('#PBS -j oe\n')
        wf.write('cd {}\n'.format(dir_path))
        wf.write(command)
        wf.write('\necho "Done" > pbs_stat_{}.log'.format(title))
    print('Submitting...')
    os.system('qsub {}.pbs > {}.scheduler 2>&1'.format(title, title))
    with open('{}.scheduler'.format(title), 'r') as scheduler:
        info_sub = scheduler.read()
        _counter_sub = 0
    while 'error' in info_sub:
        print('Submission failed, try again after 30s.')
        _counter_sub += 1
        time.sleep(30)
        os.system('qsub {}.pbs > {}.scheduler'.format(title, title))
        if _counter_sub % 5 == 0 and _counter_sub > 1:
            raise RuntimeError('Cannot submit mission')
    print('Submission succeed.')

    scheduler_idx = info_sub.split('.')[0]
    time1 = time.perf_counter()
    if is_wait:
        _counter = 0
        while is_running:
            with open('pbs_stat_{}.log'.format(title), 'r') as log:
                if log.read().split()[0] == 'Done':
                    print('waiting on PBS processing {} time: {:d} s'.format(title, int(time.perf_counter() - time1)))
                    print('Job done.')
                    is_running = False
                    break
            print('waiting on PBS processing {} time: {:d} s\r'.format(title, int(time.perf_counter() - time1)), end='')
            time.sleep(2)
            _counter += 1
            if _counter % 15 == 0:
                os.system('qstat {} > nodes_state.log'.format(host))
                with open('nodes_state.log', 'r') as nodes_state:
                    if str(scheduler_idx) in nodes_state.read():
                        continue
                    else:
                        if _counter % 2 == 0:
                            continue
                        with open('pbs_stat_{}.log'.format(title), 'r') as log:
                            if log.read().split()[0] == 'Done':
                                break
                            else:
                                raise RuntimeError('Unknown error')


def run_md(protein_dir, loop=0):
    """
    Let molecule in pdb files go into the equilibrium state through em, nvt and npt simulations. The boxes information, solvent, ions are added too.
    All initial structures and walkers have the exact same solvent number, ion number, ion type and box size. Concentration of saline is set as 0.15M.
    For this purpose, we record the information of the first structure as the template.
    """
    global num_sol, box_size, num_Na, num_Cl
    print('echo -e "1\n1\n" | gmx pdb2gmx -f %s.pdb -o processed.gro -ignh -heavyh > grompp.log 2>&1' % protein_dir)
    os.system('echo -e "1\n1\n" | gmx pdb2gmx -f %s.pdb -o processed.gro -ignh -heavyh > grompp.log 2>&1' % protein_dir)

    if (loop == 0) and not os.path.exists('../box_information.txt'):
        print('gmx editconf -f processed.gro -o newbox.gro -d 0.9 -c -bt triclinic')
        os.system(
            'gmx editconf -f processed.gro -o newbox.gro -d 0.9 -c -bt triclinic')
        print('gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        os.system(
            'gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        with open('solv.gro', 'r') as sol_gro:
            for line in sol_gro.readlines():
                info = line.split()
                # print(info)
                if len(info) == 3:
                    if all([all([j.isdigit() for j in i.split('.')]) for i in info]):
                        box_size = [float(k) + 0.10000 for k in info]

        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'SOL' and line_sp[1].isdigit():
                    num_sol = line_sp[1]
        print('Max number of solvents is:', num_sol)
        os.system('gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
        os.system(
            'echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15')
        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'NA':
                    num_Na = line_sp[1]
                if line.split()[0] == 'CL':
                    num_Cl = line_sp[1]
        with open('../box_information.txt', 'w') as box_info:
            box_info.write('num_sol={}\nbox_size={},{},{}\nnum_Na={}\nnum_Cl={}'.format(
                num_sol, box_size[0], box_size[1], box_size[2], num_Na, num_Cl))
    else:
        if os.path.exists('../box_information.txt'):
            contents = open('../box_information.txt', 'r+').readlines()
            num_sol = int(contents[0].split("=")[1])
            box_size = [float(s) for s in contents[1].split("=")[1].split(",")]
            num_Na = int(contents[2].split("=")[1])
            num_Cl = int(contents[3].split("=")[1])
        print('gmx editconf -f processed.gro -o newbox.gro -box {} {} {} -c -bt triclinic'.format(
            box_size[0], box_size[1], box_size[2]))
        os.system('gmx editconf -f processed.gro -o newbox.gro -box {} {} {} -c -bt triclinic'.format(
            box_size[0], box_size[1], box_size[2]))
        print('gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
        os.system(
            'gmx solvate -cp newbox.gro -cs spc216.gro -maxsol {} -o solv.gro -p topol.top > sol.log 2>&1'.format(
                int(num_sol)))

        with open('topol.top', 'r') as top:
            for line in top.readlines():
                line_sp = line.split()
                if line_sp == []:
                    continue
                if line.split()[0] == 'SOL' and line_sp[1].isdigit():
                    print('Max number of solvents is:', line_sp[1])

        os.system('gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
        os.system(
            'echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -np {} -nn {}'.format(
                num_Na, num_Cl))

    os.system('gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1 > grompp_em.log 2>&1')
    # os.system('gmx mdrun -deffnm em -v -nt 4')
    os.system('gmx mdrun -deffnm em -v -ntmpi 1 -nt 4')
    os.system('gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r em.gro -maxwarn 1 > grompp_nvt.log 2>&1')
    command = 'gmx mdrun -deffnm nvt -v -nt 4'
    # pbs_submit('rid_nvt_{}'.format(pdbname), command=command, dir_path=os.getcwd())
    os.system(command)
    os.system(
        'gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -r nvt.gro -maxwarn 1 > grompp_npt.log 2>&1')
    command = 'gmx mdrun -deffnm npt -v -nt 4'
    os.system(command)
    # pbs_submit('rid_npt_{}'.format(protein_dir), command=command, dir_path=os.getcwd())
    # os.system('gmx mdrun -deffnm npt -v -nt 4')
    os.system('cp topol.top topol.top.bak')


def mk_posre(dirname, pdbname):
    """
    We will add position restrain to gmx via .itp file.
    For atoms chosen, we apply a score gotten from GNNQA.1DDT, which can predict the residue division from the native structure.
    In this Version, we divided the residues in groups of 5 and ranked for every group. we chose normalized cutoff=0.65.
    All 5 residues will be chosen if group scores > 0.35, and the position restrain will be added for Ca atoms in these residues.

    2021/2/25 modified, we will use averange smooth function select CVs, instead of 'in group 5'.
    """
    window_num = 5
    qa = pickle.load(open('%s/CASP13/QA/%s.GNNQA.lDDT.pkl' % (dirname, pdbname), 'rb'))
    pdbname = pdbname + '.pdb'
    local_socre = qa[pdbname]['local']
    ave = []
    for i in range(len(local_socre)):
        if i == 0:
            ave.append((local_socre[i]+local_socre[i+1]+local_socre[i+2])/3)
        elif i == 1:
            ave.append((local_socre[i-1]+local_socre[i] +
                        local_socre[i+1]+local_socre[i+2])/4)
        elif i == len(local_socre)-2:
            ave.append(
                (local_socre[i-2]+local_socre[i-1]+local_socre[i]+local_socre[i+1])/4)
        elif i == len(local_socre)-1:
            ave.append((local_socre[i-2]+local_socre[i-1]+local_socre[i])/3)
        else:
            ave.append((local_socre[i-2]+local_socre[i-1] +
                        local_socre[i]+local_socre[i+1]+local_socre[i+2])/5)
    normalized = (ave-np.min(ave))/(np.max(ave)-np.min(ave))

    np.savetxt('normalized.txt', normalized)
    biased_ang = []
    #  <0.35 and 10% smallest
    #  <0.35 and 3 smallest
    all_index = []
    # cutoff=0.35
    normalized_list = list(normalized)
    num_smallest = 3
    min_index = map(normalized_list.index, heapq.nsmallest(num_smallest, normalized_list))
    for index in sorted(min_index):
        all_index.append(index)

    cutoff = 0.86
    for i in range(len(normalized)):
        if normalized[i] <= cutoff:
            all_index.append(i)
    biased_ang = sorted(set(all_index))
    # for i in all_index:
    #     if i<len(normalized)-1:
    #         for bb in range(i*window_num,(i+1)*window_num):
    #             biased_ang.append(bb)
    #     elif i==len(normalized)-1:
    #         for bb in range((len(normalized)-1)*window_num,len(qa[pdbname]['local'])):
    #             biased_ang.append(bb)
    print(biased_ang)

    np.savetxt('biased_res.txt', list(biased_ang), fmt='%d')
    list_biased_ang = []
    for aa in biased_ang:
        list_biased_ang.append(aa)
    os.system('cp %s/jsons/phipsi_selected.json ./' % ridkit_dir)
    replace('phipsi_selected.json', '.*selected_index.*', '    "selected_index":  %s,' % list_biased_ang)
    array_r0 = 0.6 * (1 - normalized)
    structure = 'nvt.gro'
    #   kappa=0.025      #kcal/mol/A2   *4.184*100
    # kappa=15             #kj/mol/nm2
    t_ref = md.load(structure, top=structure)
    topology = t_ref.topology
    ca_atoms = topology.select('name CA') + 1
    wf = open('posre.itp', 'w')
    wf.write('[ position_restraints ]\n;  i funct       g         r(nm)       k\n')
    for i in range(len(ca_atoms)):
        wf.write('%d    2        1          %f       TEMP\n' % (ca_atoms[i], array_r0[i // window_num]))
    wf.close()


def mk_rid(dirname, pdbname, job_dir, task="rid"):
    mol_dir = os.path.join(ridkit_dir, 'mol/', pdbname)
    print('mol_dir', mol_dir)
    print('pdbname', pdbname)
    print('dirname', dirname)
    pathlib.Path(mol_dir).mkdir(parents=True, exist_ok=True)
    os.system('cp %s/topol.top %s' % (pdbname, mol_dir))
    _r_dir = ['%s' % pdbname, '%s.GNNRefine.DAN1' % pdbname, '%s.GNNRefine.DAN2' % pdbname,
              '%s.GNNRefine.M1' % pdbname, '%s.GNNRefine.M2' % pdbname, '%s.GNNRefine.M3' % pdbname]
    for i in range(len(_r_dir)):
        os.system('cp %s/npt.gro %s/conf00%d.gro' % (_r_dir[i], mol_dir, i))
    for j in range(len(_r_dir), 8):
        os.system('cp %s/npt.gro %s/conf00%d.gro' % (_r_dir[0], mol_dir, j))
    os.system('cp %s/npt.gro %s/conf.gro' % (pdbname, mol_dir))
    os.system('cp %s/*.itp %s' % (pdbname, mol_dir))
    os.system('cp %s/mol/*.mdp %s' % (ridkit_dir, mol_dir))
    # raise RuntimeError
    os.chdir(ridkit_dir)
    os.system('python %s %s ./jsons/default_gen.json %s/%s/%s/phipsi_selected.json ./mol/%s/ -o %s/%s.run06' %
              (os.path.join(ridkit_dir, "gen.py"), task, dirname, pdbname, pdbname, pdbname, dirname, pdbname))
    os.chdir('%s/%s' % (dirname, pdbname))


def mk_score(where_rw_dir, where_evo_path, target):
    """
    generate rwplus dir in *.run dir. 3 files (calRWplus, rw.dat, scb,dat) should be in where_rw_dir.
    Args:
            where_sco_dir: containing rwplus files.
            target: name of protein.
    """
    score_dir = './{}.run06/score'.format(target)  # where they will be copied to.
    if os.path.exists(score_dir):
        shutil.rmtree(score_dir)
    os.mkdir(score_dir)
    os.system('cp -r {}/calRWplus {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/rw.dat {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/scb.dat {}'.format(where_rw_dir, score_dir))
    os.system('cp -r {}/TMscore {}'.format(os.path.join(where_rw_dir, '..'), score_dir))
    return


def main():
    parser.add_argument('target', type=str)
    parser.add_argument("TASK", type=str, help="the task", default="rid")
    parser.add_argument("-d", "--jobdir", type=str, help="job directory")
    args = parser.parse_args()
    new_target = args.target

    dirname = os.getcwd()
    pdbname = [new_target]
    num_sol = None
    box_size = []
    num_Na, num_Cl = None, None

    for pp in pdbname:
        pp = pp.strip()
        r_dir = ['%s' % pp, '%s.GNNRefine.DAN1' % pp, '%s.GNNRefine.DAN2' % pp,
                 '%s.GNNRefine.M1' % pp, '%s.GNNRefine.M2' % pp, '%s.GNNRefine.M3' % pp]
        pathlib.Path(pp).mkdir(parents=True, exist_ok=True)
        os.chdir(pp)  # at R0949/
        for num, rr in enumerate(r_dir):
            pathlib.Path(rr).mkdir(parents=True, exist_ok=True)
            os.chdir(rr)
            os.system('cp %s/CASP13/models/%s.pdb ./' % (dirname, rr))
            os.system('cp %s/mdp/* ./' % dirname)
            os.system('cp -r %s/charmm36-mar2019.ff ./' % dirname)
            change_his('./%s.pdb' % rr)
            run_md(rr, loop=num)
            replace('topol.top', '.*charmm36-mar2019.ff',
                    '#include "{}/charmm36-mar2019.ff'.format(dirname))
            mk_posre(dirname, pp)
            os.chdir('..')

        mk_rid(dirname, pp, job_dir="", task=args.TASK)
        os.chdir('..')
        mk_score(where_rw_dir='/home/dongdong/wyz/rwplus/RWplus', where_evo_path="/home/dongdong/wyz/EvoEF2-master/EvoEF2", target=pp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makr rid dir.')
    main()

