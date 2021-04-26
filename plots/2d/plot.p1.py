#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

kbT = (8.617343E-5) * 300
beta = 1.0 / kbT
f_cvt = 96.485


def load_graph(frozen_graph_filename,
               prefix='load'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=prefix,
            producer_op_list=None
        )
    return graph


def graph_op_name(graph):
    names = []
    for op in graph.get_operations():
        print(op.name)


def test_ef(sess, xx):
    graph = sess.graph

    inputs = graph.get_tensor_by_name('load/inputs:0')
    o_energy = graph.get_tensor_by_name('load/o_energy:0')
    o_forces = graph.get_tensor_by_name('load/o_forces:0')

    zero4 = np.zeros([xx.shape[0], xx.shape[1]])
    data_inputs = np.concatenate((xx, zero4), axis=1)
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run([o_energy, o_forces], feed_dict=feed_dict_test)
    return data_ret[0], data_ret[1]


def test_e(sess, xx):
    graph = sess.graph

    inputs = graph.get_tensor_by_name('load/inputs:0')
    o_energy = graph.get_tensor_by_name('load/o_energy:0')

    zero4 = np.zeros([xx.shape[0], xx.shape[1]])
    data_inputs = np.concatenate((xx, zero4), axis=1)
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run([o_energy], feed_dict=feed_dict_test)
    return data_ret[0]


def value_array_pp_0(sess, ngrid):
    xx = np.linspace(-np.pi, np.pi, (ngrid + 1))
    yy = np.linspace(-np.pi, np.pi, (ngrid + 1))
    delta = 2.0 * np.pi / ngrid

    my_grid = np.zeros(((ngrid + 1) * (ngrid + 1), 2))
    for i in range(ngrid + 1):
        for j in range(ngrid + 1):
            my_grid[i * (ngrid + 1) + j, 0] = xx[i]
            my_grid[i * (ngrid + 1) + j, 1] = yy[j]

    zz = -test_e(sess, my_grid)
    zz = zz - np.max(zz)

    return xx, yy, zz


def value_array_phi(sess, ngrid):
    xx = np.linspace(-np.pi, np.pi, (ngrid + 1))
    yy = np.linspace(-np.pi, np.pi, (ngrid + 1))
    delta = 2.0 * np.pi / ngrid
    zz = np.zeros([len(xx)])

    my_grid = np.zeros(ngrid + 1)
    zero_grid = np.zeros(ngrid + 1)
    for i in range(ngrid + 1):
        my_grid[i] = xx[i]

    for i in range(len(xx)):
        ve = test_e(sess, np.concatenate((zero_grid[:, None] + xx[i], my_grid[:, None]), axis=1))
        zz[i] = kbT * np.log(np.sum(delta * np.exp(-beta * ve)))

    zz = zz - np.max(zz)

    return xx, zz


def value_array_psi(sess, ngrid):
    xx = np.linspace(-np.pi, np.pi, (ngrid + 1))
    yy = np.linspace(-np.pi, np.pi, (ngrid + 1))
    delta = 2.0 * np.pi / ngrid
    zz = np.zeros([len(yy)])

    my_grid = np.zeros(ngrid + 1)
    zero_grid = np.zeros(ngrid + 1)
    for i in range(ngrid + 1):
        my_grid[i] = yy[i]

    for i in range(len(yy)):
        ve = test_e(sess, np.concatenate((my_grid[:, None], zero_grid[:, None] + yy[i]), axis=1))
        zz[i] = kbT * np.log(np.sum(delta * np.exp(-beta * ve)))

    zz = zz - np.max(zz)

    return yy, zz


def value_array_phipsi(sess, ngrid):
    xx = np.linspace(-np.pi, np.pi, (ngrid + 1))
    yy = np.linspace(-np.pi, np.pi, (ngrid + 1))
    xg, yg = np.meshgrid(xx, yy)
    delta = 2.0 * np.pi / ngrid
    zz0 = np.zeros([len(yy)])
    zz1 = np.zeros([len(yy)])

    my_grid = np.zeros(ngrid + 1)
    zero_grid = np.zeros(ngrid + 1)
    for i in range(ngrid + 1):
        my_grid[i] = yy[i]

    ve = test_e(sess, np.concatenate((xg.reshape((-1, 1)), yg.reshape((-1, 1))), axis=1))
    ve = ve.reshape((ngrid + 1, ngrid + 1))
    ve -= np.min(ve)
    zz0 = kbT * np.log(np.sum(delta * np.exp(-beta * ve), axis=0))
    zz1 = kbT * np.log(np.sum(delta * np.exp(-beta * ve), axis=1))

    zz0 -= np.max(zz0)
    zz1 -= np.max(zz1)

    return xx, zz0, zz1


def print_array(fname, xx, zz0, zz1, sd0, sd1):
    with open(fname, 'w') as fp:
        lx = len(xx)
        for ii in range(len(xx)):
            # fp.write ("%f %f %f %f %f \n" % (xx[ii], yy[jj], zz0[ii*lx+jj], zz1[ii*lx+jj], zz2[ii*lx+jj]))
            fp.write("%f %f %f %f %f\n" % (xx[ii], zz0[ii], zz1[ii], sd0[ii], sd1[ii]))


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--job_dir", default=".", type=str)
    parser.add_argument("-m", "--model", default=["frozen_model.pb"], type=str, nargs="*",
                        help="Frozen model file to import")
    parser.add_argument("-n", "--numb-grid", default=60, type=int,
                        help="The number of data for test")
    parser.add_argument("-o", "--output", default="fe.out", type=str,
                        help="output free energy")
    args = parser.parse_args()

    arr0 = []
    arr1 = []

    for ii in args.model:
        graph = load_graph(os.path.join(args.job_dir, ii))
        with tf.Session(graph=graph) as sess:
            xx, zz0, zz1 = value_array_phipsi(sess, args.numb_grid)
            zz0 *= f_cvt
            zz1 *= f_cvt
            arr0.append(zz0)
            arr1.append(zz1)
    avg0 = np.mean(arr0, axis=0)
    avg1 = np.mean(arr1, axis=0)

    avg0 -= np.min(avg0)
    avg1 -= np.min(avg1)

    sd0 = np.std(arr0, axis=0)
    sd1 = np.std(arr1, axis=0)
    print_array(args.output, xx, avg0, avg1, sd0, sd1)


if __name__ == '__main__':
    _main()
