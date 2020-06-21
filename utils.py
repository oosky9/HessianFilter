import SimpleITK as sitk
import numpy as np
import os
import glob

def read_mhd_and_raw(path):
    img = sitk.ReadImage(path)
    return img

def sitk2numpy(sitk_img):
    np_array = sitk.GetArrayFromImage(sitk_img) # (img(x,y,z)->numpyArray(z,y,x))
    return np_array

def numpy2sitk(np_array):
    sitk_img = sitk.GetImageFromArray(np_array)
    return sitk_img

def get_file_list(data_dir, split):
    data_dir = os.path.join(data_dir, split)
    file_list = glob.glob(data_dir + '/*.mhd')
    return file_list, len(file_list)

def write_mhd_and_raw(Data, path):
    if not isinstance(Data, sitk.SimpleITK.Image):
        print('Please check your ''Data'' class')
        return False

    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    sitk.WriteImage(Data, path, True)

    return True

def save_data_3D(path, img, name):
    
    if not os.path.isdir(path):
        os.mkdir(path)

    save_img = numpy2sitk(img)
    p =  os.path.join(path, name + ".mhd")
    write_mhd_and_raw(save_img, p)
