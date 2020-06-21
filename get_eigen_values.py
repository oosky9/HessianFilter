import numpy as np
import os
import glob
import utils
import gc

def get_file_name(path):
    n_path = path + "/*.npy"
    f_list = glob.glob(n_path)
    f_name_list = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], f_list))
    return f_list, f_name_list

def load_npy(f_list):
    for name in f_list:
        vector = np.load(name)
        yield vector

# めっちゃ時間かかる（1日くらい）．
def get_eigen(vector):
    l=len(vector)

    eigen_value = []
    for i in range(l):
        val, _ = np.linalg.eig(vector[i])
        val = val[np.argsort(np.abs(val))[::-1]]
        eigen_value.append(val)
    
    return eigen_value


def save_npy(arr, path):
    np.save(path, arr)

def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def memory_open(arr):
    del arr
    gc.collect()

def main():

    in_dir = "./result/sigma_2root2"
    val_dir = "./eigen_value/sigma_2root2"

    check_dir(val_dir)

    f_list, f_name_list = get_file_name(in_dir)

    vect_itr = load_npy(f_list)

    for i, name in enumerate(f_name_list):
        vec = next(vect_itr)
        vec = vec.reshape(-1, 3, 3)
        print("Process:{}, {}".format(i, name))
        eig_val = np.array(get_eigen(vec))
        save_npy(eig_val, os.path.join(val_dir, "eigen_value_" + name + ".npy"))
        memory_open(eig_val)

if __name__ == "__main__":
    main()