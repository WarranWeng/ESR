from tqdm import tqdm
import os
import cv2
from glob import glob


presset_size = (720, 1280)
root_data_path = '/media/wwm/wwmdisk/work/dataset/Nfs/sequences'
data_dirs = sorted([os.path.join(root_data_path, dir, '240') for dir in os.listdir(root_data_path)])

for data_dir in data_dirs:
    ori_imgs_path = sorted(glob(os.path.join(data_dir, '*.jpg')))
    print(f'{data_dir} len: {len(ori_imgs_path)}')
    for i in ori_imgs_path:
        img = cv2.imread(i, 0)
        assert img.shape[0] < img.shape[1], 'wrong H and W'
    img = cv2.imread(ori_imgs_path[0], 0)
    if img.shape[0:2] != presset_size:
        print('***********************')
        print(f'{data_dir} is NOT pre-set size: {presset_size}')

