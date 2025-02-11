# Imports
import argparse
import os

import cv2 as cv
import numpy as np
from networks import GoalNet
from utils.data import get_jpeg_files

# Program
def main(dataset:str, output:str, frame_reduction:int=1, *args, **kwargs):
    # Create output paths
    output = output or os.path.join(dataset, 'preprocessed/')
    out_scenes = os.path.join(output, 'scene_inputs/')
    os.makedirs(out_scenes, exist_ok=True)

    # load jpeg files
    jpg_files = np.array(get_jpeg_files(dataset))

    # selection for frame reduction
    selection = np.array([i % frame_reduction == 0 for i in range(len(jpg_files))])

    # redure frames
    jpg_files = jpg_files[selection]

    # load joints
    info_frames = np.load(os.path.join(dataset, 'info_frames.npz'))
    joints_2d = info_frames['joints_2d'][selection]

    # load intrinsics
    intrinsics = info_frames['intrinsics'][selection]

    # precompute values
    scene_height, scene_width = GoalNet.input_size
    heatmap_height, heatmap_width = GoalNet.output_size
    
    n_images = len(jpg_files)
    
    # begin filewise processing
    for i, filename in enumerate(jpg_files, 1):
        # print status info
        print(f"\r Processing image {i}/{n_images} ({i*100/n_images:>.2f} %)", end='')

        # load image
        img = cv.imread(filename)
        src_height, src_width = img.shape[:2]

        # resize image
        scene_img = cv.resize(img, (scene_width, scene_height))

        # Write results
        out_name = os.path.split(filename)[1]
        cv.imwrite(os.path.join(out_scenes, out_name), scene_img)
        
    print('\nDone!')
    
    # computing torso positions
    print('Scaling info frames size...')
    # Scale camera intrinsics
    intrinsics_heatmap = intrinsics.copy()
    intrinsics_heatmap[:,0,(0,2)] *= heatmap_width / src_width
    intrinsics_heatmap[:,1,(1,2)] *= heatmap_height / src_height

    intrinsics_scene = intrinsics.copy()
    intrinsics_scene[:,0,(0,2)] *= scene_width / src_width
    intrinsics_scene[:,1,(1,2)] *= scene_height / src_height

    joints_2d_scene = joints_2d.copy()
    joints_2d_scene[:,:,0] *= scene_width / src_width
    joints_2d_scene[:,:,1] *= scene_height / src_height

    joints_2d_heatmap = joints_2d.copy()
    joints_2d_heatmap[:,:,0] *= heatmap_width / src_width
    joints_2d_heatmap[:,:,1] *= heatmap_height / src_height

    np.savez(
        os.path.join(output, 'info_frames.npz'),
        joints_2d_scene=joints_2d_scene,
        joints_2d_heatmap=joints_2d_heatmap,
        joints_3d_cam = info_frames['joints_3d_cam'][selection],
        joints_3d_world = info_frames['joints_3d_world'][selection],
        world2cam_trans = info_frames['world2cam_trans'][selection],
        intrinsics_scene = intrinsics_scene,
        intrinsics_heatmap = intrinsics_heatmap
    )
    print('Done!')



# Entry point
if __name__ == '__main__':
    # fetch command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-fr', '--frame-reduction', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='../data/2020-06-09-16-09-56')
    main(**vars(parser.parse_args()))