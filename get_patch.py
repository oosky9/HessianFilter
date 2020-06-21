import numpy as np
import SimpleITK as sitk

import os
import argparse
import random


def save_filename(path, name_list):
    if not os.path.isdir(path):
        os.makedirs(path)

    f_path = path + 'filename.txt'

    with open(f_path, mode='w') as f:
        f.write('\n'.join(name_list))

def get_shift_values(d):

    random.seed(a=None, version=2)
    dx = random.uniform(-d, d)

    if dx >= 0:
        dx = int(dx + 0.5)
    elif dx < 0:
        dx = int(dx - 0.5)

    random.seed(a=None, version=2)
    dy = random.uniform(-d, d)

    if dy >= 0:
        dy = int(dy + 0.5)
    elif dy < 0:
        dy = int(dy - 0.5)

    random.seed(a=None, version=2)
    dz = int(random.uniform(-d, d) + 0.5)

    if dz >= 0:
        dz = int(dz + 0.5)
    elif dz < 0:
        dz = int(dz - 0.5)

    return dx, dy, dz

def get_cut_flag(patch_size, th, mask, x, y, z):

    patch_range = 3

    th_flag = mask[z, y, x] > th

    if th_flag:

        for k in range(-patch_range, patch_range + 1):
            for j in range(-patch_range, patch_range + 1):
                for i in range(-patch_range, patch_range + 1):
                    c_flag = mask[z, y, x] > mask[z + k, y + j, x + i] 

                    if not c_flag:
                        if i==0 and j==0 and k==0:
                            continue
                        else:
                            break
                else:
                    continue
                break
            else:
                continue
            break

        return c_flag

    else:
        return th_flag

def arg_parser():
    parser = argparse.ArgumentParser(description='py, case_name, z_size, xy_resolution')

    parser.add_argument('--patch_size', '-i1', type=int, default=32, help='patch size')

    parser.add_argument('--th', '-i2', type=int, default=50, help='threshold of hessian')

    parser.add_argument('--is_shift', '-i3', type=bool, default=False, help='shift')

    parser.add_argument('--path_w', type=str, default="D:/oosky/SIMs/data_CT/patch2/", help='output path')

    args = parser.parse_args()

    return args


def main(args):

    patch_size = args.patch_size

    w = int(patch_size/2)

    mask_path = "./output_multi/"
    img_path = "./crop_img/"

    path_w = args.path_w

    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    file_list = list(filter(lambda p: p.split(".")[-1] == "mhd", os.listdir(img_path)))

    print("Get {} dataset".format(len(file_list)))

    count = 0
    
    file_name_list = []

    for p in file_list:

        print('load data >> {}'.format(p))

        sitkimg = sitk.ReadImage(os.path.join(img_path, p))
        img = sitk.GetArrayFromImage(sitkimg)

        sitkmask = sitk.ReadImage(os.path.join(mask_path, p))
        mask = sitk.GetArrayFromImage(sitkmask)

        size = sitkimg.GetSize()

        x_size = size[0]
        y_size = size[1]
        z_size = size[2]

        img = np.reshape(img, [z_size, y_size, x_size])

        mask = np.reshape(mask, [z_size, y_size, x_size])

        d = 2.688
        dx = dy = dz = 0

        # make patch
        for z in range(z_size-1):
            for y in range(y_size-1):
                for x in range(x_size-1):
                    
                    try:
                        cut_flag = get_cut_flag(patch_size, args.th, mask, x, y, z)
                    except:
                        continue

                    if cut_flag:
                        
                        if args.is_shift:
                            dx, dy, dz = get_shift_values(d)
                            
                        patch = img[z-w+dz:z+w+dz, y-w+dy:y+w+dy, x-w+dx:x+w+dx]

                        try:
                            patch = patch.reshape([patch_size, patch_size, patch_size])
                        except:
                            continue
                            

                        if np.all(patch!=1000) and np.max(patch) < 200 and np.min(patch) > -1200:

                            eudt_image = sitk.GetImageFromArray(patch)
                            eudt_image.SetSpacing(sitkimg.GetSpacing())
                            eudt_image.SetOrigin(sitkimg.GetOrigin())

                            if args.is_shift:
                                sitk.WriteImage(eudt_image, os.path.join(path_w , "patch_{}_{}_{}.mhd".format(x+dx, y+dy, z+dz)))
                                n = os.path.join(path_w ,"patch{}_{}_{}.mhd".format(x+dx, y+dy, z+dz))
                                file_name_list.append(n)
                            
                            else:
                                sitk.WriteImage(eudt_image, os.path.join(path_w, "patch_{}.mhd".format(count)))
                                n = os.path.join(path_w, "patch_{}.mhd".format(count))
                                file_name_list.append(n)
                            
                            count += 1
                            if count % 100 == 0:
                                print(count)
    
    save_filename(path_w, file_name_list)


if __name__ == '__main__':
    args = arg_parser()
    main(args)