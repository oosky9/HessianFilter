import numpy as np
import utils
import os
import glob

def get_file_name(path):
    n_path = path + "/*.npy"
    f_list = glob.glob(n_path)
    f_name_list = list(map(lambda path: path.split(os.sep)[-1].split(".")[0].replace("linearity_", ''), f_list))
    return f_list, f_name_list

def load_npy(f_list):
    for name in f_list:
        vector = np.load(name)
        yield vector

def main():

    in_dir = "./linearity_multi"
    out_dir = "./output_multi"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    f_path_list, f_name_list = get_file_name(in_dir)

    img_itr = load_npy(f_path_list)

    for i, _ in enumerate(f_path_list):
        np_img = next(img_itr)
        print("Process:{}, {}".format(i, f_name_list[i]))
        utils.save_data_3D(out_dir, np_img, f_name_list[i])


if __name__ == "__main__":
    main()