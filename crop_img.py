import numpy as np
import utils
import glob
import os

def get_file_name(path):
    n_path = path + "/*.mhd"
    f_list = glob.glob(n_path)
    f_name_list = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], f_list))
    return f_list, f_name_list

def load_image(f_list):
    for p in f_list:
        temp_sitk = utils.read_mhd_and_raw(p)
        img = utils.sitk2numpy(temp_sitk)
        yield img

def crop_3D(img):
    crop = img[:400, 50:400, 25:475]
    print(crop.shape)
    return crop

def save_img(path, img, name):
    print("---save image---")
    utils.save_data_3D(path, img, name)

    
def main():

    out_dir = "./crop_img"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    f_list, f_name_list = get_file_name("../LungCT_OriginalDATA/Tokushima/Lung")
    
    img_itr = load_image(f_list)

    for i, name in enumerate(f_name_list):
        print("Process:{}, {}".format(i, name))
        lung_img = next(img_itr)
        tmp = crop_3D(lung_img)
        save_img(out_dir, tmp, name)

if __name__ == "__main__":
    main()
