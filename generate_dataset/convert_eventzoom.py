import os
import cv2
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm
import esim_py
import random
import shutil
import pandas as pd
# local modules
from tools.event_packagers import hdf5_event_packager


config = {
    'Cp_init': 0.1,
    'Cn_init': 0.1,
    'refractory_period': 1e-4,
    'log_eps': 1e-3,
    'use_log':True,
    'CT_range': [0.1, 0.1],
    'max_CT': 0.7,
    'min_CT': 0.01,
    'mu': 1,
    'sigma': 0.1,
    'H': 720,
    'W': 1280,
    'fps': 240,
}


def write_img(img: np.ndarray, idx: int, imgs_dir: str):
    assert os.path.isdir(imgs_dir)
    path = os.path.join(imgs_dir, "%05d.png" % idx)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, img)


def write_timestamps(timestamps: list, timestamps_filename: str):
    with open(timestamps_filename, 'w') as t_file:
        t_file.writelines([str(t) + '\n' for t in timestamps])


def write_config(config_path):
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value} \n')


def prepare_output_dir(src_dir: str, dest_dir: str):
    # Copy directory structure.
    def ignore_files(directory, files):
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    shutil.copytree(src_dir, dest_dir, ignore=ignore_files)


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', default='/media/wwm/wwmdisk/work/dataset/Nfs/eventzoom_data')
    parser.add_argument('--path_to_h5', default='/media/wwm/wwmdisk/work/dataset/Nfs/eventzoom_data/h5')
    args = parser.parse_args()

    return args


def transform_events(events):
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    p[p==0] = -1
    return np.stack([x, y, t, p], axis=1)


if __name__ == '__main__':
    flags = get_flags()

    root_data_path = flags.root_data_path
    path_to_h5 = flags.path_to_h5

    path_to_hr = os.path.join(root_data_path, 'data/ev_hr')
    path_to_lr = os.path.join(root_data_path, 'data/ev_lr_1')
    path_to_llr = os.path.join(root_data_path, 'data/ev_llr_1')
    assert os.path.exists(path_to_hr)
    assert os.path.exists(path_to_lr)
    assert os.path.exists(path_to_llr)

    if not os.path.exists(path_to_h5):
        os.makedirs(path_to_h5, exist_ok=False)

    hr_dirs = sorted(glob(os.path.join(path_to_hr, '*.txt')))
    lr_dirs = sorted(glob(os.path.join(path_to_lr, '*.txt')))
    llr_dirs = sorted(glob(os.path.join(path_to_llr, '*.txt')))

    for hr_dir, lr_dir, llr_dir in zip(hr_dirs, lr_dirs, llr_dirs):
        print(f'\n processing {hr_dir, lr_dir, llr_dir}')
        assert os.path.basename(hr_dir) == os.path.basename(lr_dir) == os.path.basename(llr_dir)

        ori_hr_ev = pd.read_csv(hr_dir, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.float64},
                           engine='c',
                           skiprows=1, nrows=None, memory_map=True).values
        hr_events = transform_events(ori_hr_ev)
        ori_lr_ev = pd.read_csv(lr_dir, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.float64},
                           engine='c',
                           skiprows=1, nrows=None, memory_map=True).values
        lr_events = transform_events(ori_lr_ev)
        ori_llr_ev = pd.read_csv(llr_dir, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.float64},
                           engine='c',
                           skiprows=1, nrows=None, memory_map=True).values
        llr_events = transform_events(ori_llr_ev)

        # save events to h5
        h5_filename = os.path.basename(hr_dir).split('.')[0] + '.h5'
        ep = hdf5_event_packager(os.path.join(path_to_h5, h5_filename))
        ep.package_events('ori', hr_events[:, 0], hr_events[:, 1], hr_events[:, 2], hr_events[:, 3])
        ep.package_events('down2', lr_events[:, 0], lr_events[:, 1], lr_events[:, 2], lr_events[:, 3])
        ep.package_events('down4', llr_events[:, 0], llr_events[:, 1], llr_events[:, 2], llr_events[:, 3])

        ep.add_data([124, 222])

    print('all {} files are done!'.format(len(hr_dirs)))