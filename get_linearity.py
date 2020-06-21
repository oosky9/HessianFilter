import tensorflow as tf
import numpy as np

import os
import glob
import gc

def load_npy(f_list):
    for name in f_list:
        vector = np.load(name)
        yield vector

def get_file_name(path):
    n_path = path + "/*.npy"
    f_list = glob.glob(n_path)
    f_name_list = list(map(lambda path: path.split(os.sep)[-1].split(".")[0].replace("eigen_value_", ''), f_list))
    return f_list, f_name_list

def save_npy(arr, path):
    np.save(path, arr)

def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def memory_open(arr):
    del arr
    gc.collect()

def main():

    sig_1 = 1.0        # 1**2
    sig_root2 = 2.0    # root2**2
    sig_2 = 4.0        # 2**2
    sig_2root2 = 8.0   # 2root2**2

    data_lst = ["sigma_1", "sigma_root2", "sigma_2", "sigma_2root2"]

    in_dir = "./eigen_value"
    val_dir = "./linearity_multi"

    check_dir(val_dir)

    sig1_list, f_name_list = get_file_name(os.path.join(in_dir, data_lst[0]))
    sigr2_list  = list(map(lambda p: os.path.join(os.path.join(in_dir, data_lst[1]), "eigen_value_" + p + ".npy"), f_name_list))
    sig2_list   = list(map(lambda p: os.path.join(os.path.join(in_dir, data_lst[2]), "eigen_value_" + p + ".npy"), f_name_list))
    sig2r2_list = list(map(lambda p: os.path.join(os.path.join(in_dir, data_lst[3]), "eigen_value_" + p + ".npy"), f_name_list))

    sig1_itr   = load_npy(sig1_list)
    sigr2_itr  = load_npy(sigr2_list)
    sig2_itr   = load_npy(sig2_list)
    sig2r2_itr = load_npy(sig2r2_list)

    vect = tf.placeholder(tf.float32, shape=[None, 3])
    t_vect = tf.placeholder(tf.float32, shape=[4, None])

    s_1 = tf.constant(sig_1, dtype=tf.float32)
    s_r2 = tf.constant(sig_root2, dtype=tf.float32)
    s_2 = tf.constant(sig_2, dtype=tf.float32)
    s_2r2 = tf.constant(sig_2root2, dtype=tf.float32)
    
    linear = tf.subtract(vect[:, 2], vect[:, 1]) # lambda_3 - lambda_2

    linear_s1 = tf.multiply(linear, s_1)
    linear_sr2 = tf.multiply(linear, s_r2)
    linear_s2 = tf.multiply(linear, s_2)
    linear_s2r2 = tf.multiply(linear, s_2r2)

    max_val = tf.reduce_max(t_vect, axis=0)

    with tf.Session() as sess:

        for i, name in enumerate(f_name_list):
            print("Process:{}, {}".format(i, name))

            lin_lst = []

            vec = next(sig1_itr)
            val = sess.run(linear_s1, feed_dict={vect: vec})
            lin_lst.append(val)
            vec = next(sigr2_itr)
            val = sess.run(linear_sr2, feed_dict={vect: vec})
            lin_lst.append(val)
            vec = next(sig2_itr)
            val = sess.run(linear_s2, feed_dict={vect: vec})
            lin_lst.append(val)
            vec = next(sig2r2_itr)
            val = sess.run(linear_s2r2, feed_dict={vect: vec})
            lin_lst.append(val)

            lin_max = sess.run(max_val, feed_dict={t_vect: np.array(lin_lst)})
            lin_max = lin_max.reshape(400, 350, 450)
            save_npy(lin_max, os.path.join(val_dir, "linearity_" + name + ".npy"))

            memory_open(lin_lst)
            memory_open(lin_max)


if __name__ == "__main__":
    main()