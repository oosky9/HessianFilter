import numpy as np
import utils
import os
import SimpleITK as sitk
import sys

def load_image(path, split):
    f_list, _ = utils.get_file_list(path, split)

    f_name_list = list(map(lambda path: path.split(os.sep)[-1], f_list))

    img=[]

    for p in f_list:
        temp_sitk = utils.read_mhd_and_raw(p)
        temp_np   = utils.sitk2numpy(temp_sitk)
        img.append(temp_np)

    return f_name_list, img

def main():

    data_dir = "../LungCT_OriginalDATA/Tokushima"
    out_dir  = "../LungCT_OriginalDATA/Tokushima/Lung"

    ct_list, ct_img   = load_image(data_dir, "CT")
    mask_list, mask_img = load_image(data_dir, "Label")

    assert ct_list == mask_list
    
    for ct, mask, name in zip(ct_img, mask_img, ct_list):
        flag = (mask != 1) & (mask != 2)

        ct[flag] = 1000
        
        print("Saving ==>> {}".format(name))
        utils.save_data_3D(out_dir, ct, name)


if __name__ == "__main__":
    main()




