import pickle
def unpickle(file):
    with open(file,"rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
        return dict

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

import glob
import numpy as np
import os
import cv2

Test_path_list = glob.glob("E:\\imooc\\pytorch_course\\06\\Cifar10\\test_batch")
save_path = "E:\\imooc\\pytorch_course\\06\\Cifar10\\Test02"
print(Test_path_list)

for l in Test_path_list:
    l_dict = unpickle(l)
    print(l_dict.keys())

    for im_index, im_data in enumerate(l_dict[b'data']):

        im_label = l_dict[b'labels'][im_index]
        im_name = l_dict[b'filenames'][im_index]
        # print(im_label)
        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3,32,32])
        im_data = np.transpose(im_data, [1,2,0])

        # print(im_name.decode("utf-8"))
        # print(im_label_name)
        # cv2.imshow("im_data", cv2.resize(im_data,(200,200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}\\{}".format(save_path,im_label_name)):
            os.mkdir("{}\\{}".format(save_path, im_label_name))

        cv2.imwrite("{}\\{}\\{}".format(save_path,im_label_name,im_name.decode("utf-8")), im_data)
