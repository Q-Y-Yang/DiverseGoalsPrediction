# Imports
import argparse
import json
import logging
import os
import cv2 as cv
import numpy as np
from networks import GoalNet
## DEBUG
import matplotlib.pyplot as plt

# Constants
SOURCE = '/media/DATA/PROX'
OUTPUT = None
JOINT_KEYS = [
    'SpineBase',
    'SpineMid',
    'Neck',
    'Head',
    'ShoulderLeft',
    'ElbowLeft',
    'WristLeft',
    'HandLeft',
    'ShoulderRight',
    'ElbowRight',
    'WristRight',
    'HandRight',
    'HipLeft',
    'KneeLeft',
    'AnkleLeft',
    'FootLeft',
    'HipRight',
    'KneeRight',
    'AnkleRight',
    'FootRight',
    'SpineShoulder',
]

# Functons
def load_camrea_intrinsics(path:str) -> np.ndarray:
    with open(path, mode='r') as json_file:
        return np.array(json.load(json_file)['camera_mtx'])

def load_joints(path:str) -> np.ndarray:
    j = None
    with open(path, mode='r') as file:
        j = json.load(file)['Bodies']
        
    if len(j) < 1:
        return None
    j = j[0]

    joints = np.array(tuple(
        v['Position'] 
        for k, v in j['Joints'].items()
        if k in JOINT_KEYS
    ))
    return joints*(-1,1,-1)

def load_transforms(path:str) -> tuple[np.ndarray, np.ndarray]:
    cam2world = None
    with open(path, mode='r') as file:
        cam2world = np.array(json.load(file))

    return cam2world, np.linalg.inv(cam2world)

def homogenous_transform(p, t):
    return np.append(p,1).dot(t)[:-1]


# Program
def main(source:str=SOURCE, output:str=OUTPUT, *args, **kwargs):
    # logging setup
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('main')

    if output is None:
        output = source

    # DEBUG desiged formats
    x = np.load('../data/FPS-5/2020-06-09-16-09-56/preprocessed/info_frames.npz')
    logger.debug(f"Required shape - Intrinsics: {x['intrinsics_scene'].shape[1:]}")
    logger.debug(f"Required shape - TWordl2Cam: {x['world2cam_trans'].shape[1:]}")

    # Getting camera intrinsics
    intrinsics_file = os.path.join(source, 'calibration/Color.json')
    if not os.path.exists(intrinsics_file):
        logger.fatal(f"Calibation not found ({intrinsics_file})")
        return -1

    intrinsics = load_camrea_intrinsics(intrinsics_file)

    recordings_dir = os.path.join(source, 'recordings')
    if not os.path.exists(recordings_dir):
        logger.fatal(f"Recordings not found ({recordings_dir})")
        return -1


    # get dimensions
    sample_rec = os.path.join(recordings_dir, os.listdir(recordings_dir)[0], 'Color')
    sample_frame = cv.imread(os.path.join(sample_rec, os.listdir(sample_rec)[0]))
    src_height, src_width = sample_frame.shape[:2]
    scene_height, scene_width = GoalNet.input_size

    # scale intrinsics
    intrinsics[0,(0,2)] *= scene_width / src_width
    intrinsics[1,(1,2)] *= scene_height / src_height

    # process recordings
    for name in sorted(os.listdir(recordings_dir)):
        # get cam2world transform
        transform_file = os.path.join(
            source, 'cam2world', name.split('_')[0]+'.json')
        cam2world, world2cam = load_transforms(transform_file)
        
        # generate base paths
        path = os.path.join(recordings_dir, name)
        frames_dir = os.path.join(path, 'Color')
        joints_dir = os.path.join(path, 'Skeleton')

        # get frame files
        frame_files = sorted(os.listdir(frames_dir))

        # arrays
        n = len(frame_files)
        d = int(np.log10(n)) + 1
        j = len(JOINT_KEYS)
        joints_3d_world = np.zeros((n, j, 3))
        joints_2d_scene = np.zeros((n, j, 2))
        world2cam_trans = np.zeros((n, 4, 4))
        intrinsics_scene = np.zeros((n, 3, 3))
        
        # output
        out_dir = os.path.join(output, 'recordings', name, 'preprocessed')
        out_frames_dir = os.path.join(out_dir, 'scene_inputs')
        os.makedirs(out_dir)
        os.makedirs(out_frames_dir)
        for i, (file_name, extension) in enumerate(map(os.path.splitext, frame_files)):
            # generate file names
            frame_file = os.path.join(frames_dir, file_name + extension)
            joints_file = os.path.join(joints_dir, file_name + '.json')
            print(f"\r[{i+1}/{n}] {frame_file}", end='')

            # load joints
            joints_3d_cam = load_joints(joints_file)
            if joints_3d_cam is None:
                continue

            joints_3d_world[i] = np.array(tuple(
                map(lambda p: homogenous_transform(p, cam2world), joints_3d_cam)
            ))
            
            # joints 2d
            joints_2d_scene[i] = np.array(tuple(
                (x/k, y/k)
                for x, y, k, in map(intrinsics.dot, joints_3d_cam)
            ))

            world2cam_trans[i] = world2cam
            intrinsics_scene[i] = intrinsics

            # load frames
            frame = cv.imread(frame_file)
            frame = cv.resize(frame, (scene_width, scene_height))

            # gegerate frame file name
            outfile = os.path.join(
                out_frames_dir, 
                str(i).rjust(d, '0')+extension)
            cv.imwrite(outfile, frame)
        print()
        np.savez(
            os.path.join(out_dir, 'info_frames.npz'),
            joints_2d_scene=joints_2d_scene,
            joints_3d_world = joints_3d_world,
            world2cam_trans = world2cam_trans,
            intrinsics_scene = intrinsics_scene,
        )

    return 0

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default=SOURCE)
    parser.add_argument('-o', '--output', type=str, default=OUTPUT)

    exit(main(**vars(parser.parse_args())))