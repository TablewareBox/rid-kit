#!/usr/bin/env python3
import os
import glob
import argparse
import mdtraj as md
import numpy as np
import pickle
import pathlib
import re
import shutil
import time
import heapq

"""
This file can make rid dir for given molecular.
Please modify 'protein_dir'.
Last update date: 2021/2/24
Author: Dongdong Wang, Yanze Wang.
"""

ridkit_dir = "/data1/ddwang/cjh/loopOpt_test/rid-kit"


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


def change_his(protein_dir):
    """
    This function can change all the HIS residues to HSD residues in pdbfile(protein_dir). 
    it's used to meet the need of force field file. Some pdbs have different H+ sites on HIS residues, varying from HSD to HSP.
    """
    with open(protein_dir, 'r') as pdb:
        ret = pdb.read()
    ret = ret.replace('HIS', 'HSD')
    with open(protein_dir, 'w') as pdb:
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
                    print('waiting on PBS processing {} time: {:d} s'.format(
                        title, int(time.perf_counter() - time1)))
                    print('Job done.')
                    is_running = False
                    break
            print('waiting on PBS processing {} time: {:d} s\r'.format(
                title, int(time.perf_counter() - time1)), end='')
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
        os.system(
            'gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
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

        os.system(
            'gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
        os.system(
            'echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -np {} -nn {}'.format(
                num_Na, num_Cl))

    os.system(
        'gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1 > grompp_em.log 2>&1')
    # os.system('gmx mdrun -deffnm em -v -nt 4')
    os.system('gmx mdrun -deffnm em -v -ntmpi 1 -nt 4')
    os.system(
        'gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r em.gro -maxwarn 1 > grompp_nvt.log 2>&1')
    command = 'gmx mdrun -deffnm nvt -ntmpi 1 -v -nt 4'
    # pbs_submit('rid_nvt_{}'.format(protein_dir), command=command, dir_path=os.getcwd())
    os.system(command)
    os.system(
        'gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -r nvt.gro -maxwarn 1 > grompp_npt.log 2>&1')
    command = 'gmx mdrun -deffnm npt -ntmpi 1 -v -nt 4'
    os.system(command)
    # pbs_submit('rid_npt_{}'.format(protein_dir), command=command, dir_path=os.getcwd())
    # os.system('gmx mdrun -deffnm npt -v -nt 4')
    os.system('cp topol.top topol.top.bak')


def mk_posre(dirname, job_dir, loop_res=[], flat_bottom=-1, chain_id=0):
    """
    We will add position restrain to gmx via .itp file.
    For atoms chosen, we apply a score gotten from GNNQA.1DDT, which can predict the residue division from the native structure. 
    In this Version, we divided the residues in groups of 5 and ranked for every group. we chose normalized cutoff = 0.35.
    All 5 residues will be chosen if group scores > 0.35, and the position restrain will be added for Ca atoms in these residues.

    2021/2/25 modified, we will use averange smooth function select CVs, instead of 'in group 5'.
    """
    job_dir = os.path.abspath(job_dir)
    os.chdir(job_dir)

    biased_ang = sorted(set(loop_res))
    print(biased_ang)
    np.savetxt('biased_res.txt', list(biased_ang), fmt='%d')
    list_biased_ang = []
    for aa in biased_ang:
        list_biased_ang.append(aa)
    os.system('cp %s/jsons/phipsi_selected.json ./' % ridkit_dir)
    replace('phipsi_selected.json', '.*selected_index.*',
            '    "selected_index":  %s,' % list_biased_ang)

    structure_pdb = 'conf000/conf000.pdb'
    structure = 'conf000/nvt.gro'
    #   kappa=0.025      #kcal/mol/A2   *4.184*100
    # kappa=15             #kj/mol/nm2

    cmd_make_ndx = '''\
gmx make_ndx -f %s -o index.ndx << EOF
r%d-%d
name 19 Loop
r%d-%d & 4
name 20 Loop-Backbone
r%d-%d & 6
name 21 Loop-MainChain
q
EOF''' % (structure,
          list_biased_ang[0], list_biased_ang[-1],
          list_biased_ang[0], list_biased_ang[-1],
          list_biased_ang[0], list_biased_ang[-1])
    os.system(cmd_make_ndx)

    t_ref = md.load(structure, top=structure_pdb)
    t_select = md.load(structure, top=structure)
    topology = t_ref.topology
    top_select = t_select.topology

    atoms_loop = [[] for _ in range(topology.n_chains)]
    atoms_loopCA = [[] for _ in range(topology.n_chains)]

    posre_flat_bottom = ['[ position_restraints ]\n;  i funct       g         r(nm)       k\n' for _ in range(topology.n_chains)]

    for ch in range(topology.n_chains):
        atoms_before = 0 if ch == 0 else list(topology.chain(ch - 1).atoms)[-1].index + 1
        if ch != chain_id:
            continue
        for res in loop_res:
            res_atoms_global = list(top_select.select('(mass 2.0 to 90) and (residue %d) and (chainid %d)' % (res, ch)) + 1)
            res_atoms = [(atom - atoms_before) for atom in res_atoms_global]
            atoms_loop[ch] += res_atoms

            ca_atoms_global = list(top_select.select('(name CA) and (residue %d) and (chainid %d)' % (res, ch)) + 1)
            ca_atoms = [(atom - atoms_before) for atom in ca_atoms_global]
            atoms_loopCA[ch] += ca_atoms
            for i in range(len(ca_atoms)):
                posre_flat_bottom[ch] += '%d    2        1          %f       TEMP\n' % (ca_atoms[i], max(0, flat_bottom))

    atoms_loop = [[str(atom) for atom in chain] for chain in atoms_loop]
    print(atoms_loop)

    for root, dirs, files in os.walk(job_dir):
        conf_dirs = [dir for dir in dirs if dir.startswith("conf")]
        break

    os.chdir(conf_dirs[0])
    posre_files = glob.glob("posre*.itp")
    posre_files.sort()
    os.chdir(job_dir)

    assert len(posre_files) == topology.n_chains

    for ch in range(topology.n_chains):
        for conf_dir in conf_dirs:
            os.chdir(conf_dir)
            wf = open(posre_files[ch], 'r+')
            lines = wf.readlines()
            wf.seek(0, 0)

            posre_all = ''

            for line in lines:
                numbers = line.lstrip().split()
                if numbers == []:
                    posre_all += line
                    continue
                atom_str = numbers[0]
                if atom_str not in atoms_loop[ch]:
                    posre_all += line
            if flat_bottom != -1:
                posre_all += posre_flat_bottom
            wf.write(posre_all)
            wf.close()
            os.chdir(job_dir)

        wf = open(posre_files[ch], 'w')
        wf.write(posre_all)
        wf.close()


def mk_rid(dirname, pdbname, job_dir, task="rid"):
    mol_dir = os.path.join(job_dir, 'mol/', pdbname)
    # mol_dir='%s/rid-kit/mol/%s'+protein_dir
    print('mol_dir', mol_dir)
    print('pdbname', pdbname)
    print('dirname', dirname)
    pathlib.Path(mol_dir).mkdir(parents=True, exist_ok=True)

    os.chdir(job_dir)
    os.system('cp %s/topol.top %s' % ("conf000", mol_dir))
    for root, dirs, files in os.walk(job_dir):
        conf_dirs = [dir for dir in dirs if dir.startswith("conf")]
        break

    for i in range(len(conf_dirs)):
        os.system('cp %s/npt.gro %s/conf00%d.gro' % (conf_dirs[i], mol_dir, i))
    for j in range(len(conf_dirs), 8):
        os.system('cp %s/npt.gro %s/conf00%d.gro' % (conf_dirs[0], mol_dir, j))
    os.system('cp %s/npt.gro %s/conf.gro' % ("conf000", mol_dir))
    os.system('cp posre*.itp %s/' % (mol_dir))
    os.system('cp index.ndx %s/index.ndx' % (mol_dir))
    os.system('cp %s/mol/*.mdp %s' % (ridkit_dir, mol_dir))
    os.chdir(ridkit_dir)
    os.system('python %s %s ./jsons/default_gen.json %s/phipsi_selected.json %s -o %s' %
              (os.path.join(ridkit_dir, "gen.py"), task, job_dir, mol_dir, os.path.join(job_dir, "run06")))
    os.chdir(job_dir)


def mk_rwplus(where_rw_dir, target):
    """
    generate rwplus dir in *.run dir. 3 files (calRWplus, rw.dat, scb,dat) should be in where_rw_dir.
    Args:
            where_rw_dir: containing rwplus files.
            target: name of protein.
    """
    rw_dir = './{}.run06/rwplus'.format(
        target)  # where they will be copied to.
    print(rw_dir)
    if os.path.exists(rw_dir):
        shutil.rmtree(rw_dir)
    os.mkdir(rw_dir)
    os.system('cp -r {}/calRWplus {}'.format(where_rw_dir, rw_dir))
    os.system('cp -r {}/rw.dat {}'.format(where_rw_dir, rw_dir))
    os.system('cp -r {}/scb.dat {}'.format(where_rw_dir, rw_dir))
    return


pdbname = 'CASP11-T0818-K92-P111'
job_dir = os.path.join(os.getcwd(), pdbname)

protein_dir = os.path.join(os.getcwd(), '../predictions/CASP11/T0818/K92_P111/disgro')
protein_files = [("disgro_%d_whole.pdb" % i) for i in range(1, 9)]

loop = [92, 111]

dirname = os.getcwd()
num_sol = None
box_size = []
num_Na, num_Cl = None, None


def main():
    global job_dir, loop
    files = [os.path.join(protein_dir, name) for name in protein_files]
    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)
    print(job_dir)
    job_dir = os.path.abspath(job_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("TASK", type=str, help="the task", default="rid")
    parser.add_argument("-d", "--jobdir", type=str, help="job directory", default=job_dir)
    parser.add_argument("-m", "--mol", default=files, type=str, help="the mol dir", nargs="*")
    parser.add_argument("-l", "--loop", default=loop, type=int, help="loop range", nargs="*")
    parser.add_argument("-c", "--chain", default=0, type=int, help="chain id")

    args = parser.parse_args()
    job_dir = os.path.abspath(args.jobdir)
    files = [os.path.abspath(file) for file in args.mol]
    loop = args.loop
    loop_res = list(range(loop[0], loop[1]+1))

    os.makedirs(job_dir, exist_ok=True)
    os.chdir(job_dir)  # at R0949/

    for num, file in enumerate(files):
        task_name = "conf00%d" % (num)
        os.makedirs(os.path.join(job_dir, task_name), exist_ok=True)
        os.chdir(os.path.join(job_dir, task_name))

        os.system('cp %s ./%s.pdb' % (file, task_name))
        os.system('cp %s/mdp/* ./' % dirname)
        os.system('cp -r %s/charmm36-mar2019.ff ./' % dirname)
        change_his('./%s.pdb' % task_name)

        run_md(task_name, loop=num)
        replace('topol.top', '.*charmm36-mar2019.ff',
                '#include "{}/charmm36-mar2019.ff'.format(dirname))
        os.chdir(job_dir)
    mk_posre(dirname, job_dir=job_dir, loop_res=loop_res, chain_id=args.chain)
    print(os.getcwd())
    mk_rid(dirname=dirname, pdbname=pdbname, job_dir=job_dir, task=args.TASK)
    os.chdir('..')


if __name__ == '__main__':
    main()
