import tensorflow as tf
import numpy as np

import gc
import glob
import os

import utils

def get_hessian_filter(x, y, z, sigma):
    HesG = 1/((np.sqrt(2*np.pi))**3*sigma**7) * np.exp(-(x**2+y**2+z**2)/(2*sigma**2)) \
        *np.array([[x**2-sigma**2, x*y, x*z], [x*y, y**2-sigma**2, y*z], [z*x, z*y, z**2-sigma**2]], dtype=np.float32)

    HesG = HesG.flatten()
    return HesG

def get_hessian_list(sigma):
    x = list(np.linspace(-5, 5, 11))
    y = list(np.linspace(-5, 5, 11))
    z = list(np.linspace(-5, 5, 11))

    filter_list = []

    for z_i in z:
        for y_i in y:
            for x_i in x:
                filter_list.append(get_hessian_filter(x_i, y_i, z_i, sigma))
    np_filters = np.array(filter_list).reshape(11, 11, 11, 1, 9)

    return np_filters

def tf_conv3D(input, kernel):
    conv = tf.nn.conv3d(input, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
    return conv
    
def load_image(f_list):
    for p in f_list:
        temp_sitk = utils.read_mhd_and_raw(p)
        img = utils.sitk2numpy(temp_sitk)
        yield img

def get_file_name(path, split):
    f_list, _ = utils.get_file_list(path, split)
    f_name_list = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], f_list))
    return f_list, f_name_list

def memory_free(arr):
    del arr
    gc.collect()
    

def main():
    sigma = 2.0*np.sqrt(2.0)

    out_dir = "./result/sigma_2root2"
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    f_list, f_name_list = get_file_name("./crop_img", "")

    img_itr = load_image(f_list)

    hes_init = get_hessian_list(sigma=sigma)

    input_img = tf.placeholder(tf.float32, shape=[None, 400, 350, 450, 1])
    kernel = tf.constant(hes_init, tf.float32, shape=[11, 11, 11, 1, 9])

    filtering = tf_conv3D(input_img, kernel)

    with tf.Session() as sess:

        for i, name in enumerate(f_name_list):
            print("Process:{}, {}".format(i, name))
            f_name = os.path.join(out_dir, name + ".npy") 
            lung_img = next(img_itr)
            lung_img = lung_img.reshape(1, 400, 350, 450, 1)
            out_vect = sess.run(filtering, feed_dict={input_img: lung_img})
            np.save(f_name, out_vect)
            memory_free(out_vect)

if __name__ == "__main__":
    main()


    
