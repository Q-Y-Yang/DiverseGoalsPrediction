# Imports
import os
import cv2 as cv
import numpy as np
#from networks import GoalNet
from utils.probabilistic import fill_gaussian, fill_multi_gaussian

# Constants
TORSO_IDX = 10

def clamp_number(num,limit_min, limit_max):
  return max(min(num, max(limit_min, limit_max)), min(limit_min, limit_max))


#Function to create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint.
def gaussian(xL, yL, H, W, sigma):
    # crate blank heatmap
    heatmap = np.zeros((H, W), dtype=float)

    #heatmap = [np.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]

    for r in range(H):
        for c in range(W):
            heatmap[r, c] = np.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) 
    #channel = np.array(channel, dtype=np.float32)
    #heatmap = np.reshape(heatmap, newshape=(H, W))

    return heatmap


# Functions
def create_heatmap(width, height, mu, sigma):
    heatmap = gaussian(clamp_number(mu[0], 0+1, width - 1), clamp_number(mu[1], 0 +1, height - 1), height, width, sigma)

    return heatmap

def get_jpeg_files(path:str):
    return sorted(
        os.path.join(path, file_name)
        for file_name in os.listdir(path)
        if os.path.splitext(file_name)[-1].lower() == '.jpg'
    )
    
def get_preprocessed_subdirectories(base_dir:str):
    return [
        preprocessed_dir
        for preprocessed_dir in (
            os.path.join(base_dir, direcotry, 'preprocessed')
            for direcotry in os.listdir(base_dir)
        )
        if os.path.exists(preprocessed_dir)
    ]
    
def np_batch_mean_pool(a, n):
    N, H, W = a.shape
    Hn = H // n
    Wn = W // n
    return a[:,:Hn*n, :Wn*n].reshape(N, Hn, n, Wn, n).mean(axis=(2, 4))

def world2image(p_world, world2cam, intrinsics):
    p = np.append(p_world, 1)
    p_cam = np.matmul(p, world2cam)[:-1]
    x, y, k = np.matmul(intrinsics, p_cam)
    return np.array([x, y]) / k

# Load data with low and upper limits, and also sampling by setting up 'step'.
def load_preprocessed_sampling(path:str, low_limit:int=None, up_limit:int=None, step:int=None):
    # load torso positions
    info_frames = np.load(os.path.join(path, 'info_frames.npz'))
    joints = info_frames['joints_3d_world']

    # check if limit is set
    up_limit = min(up_limit or joints.shape[0], joints.shape[0])

    # load scenes
    scenes = np.array([
        cv.imread(file)
        for file in get_jpeg_files(os.path.join(path, 'scene_inputs/'))[low_limit:up_limit:step]
    ]).swapaxes(3, 2).swapaxes(2, 1)

    # load heatmaps
    heatmaps = np.array([
        cv.imread(file)
        for file in get_jpeg_files(os.path.join(path, 'heatmaps/'))[low_limit:up_limit:step]
    ]).swapaxes(3, 2).swapaxes(2, 1)

    goal_heatmaps = np.array([
        cv.imread(file)
        for file in get_jpeg_files(os.path.join(path, 'goal_heatmaps/'))[low_limit:up_limit:step]
    ]).swapaxes(3, 2).swapaxes(2, 1)
    print("len of goal heatmap", len(goal_heatmaps))
    return scenes, heatmaps, goal_heatmaps

def load_preprocessed(path:str, limit:int=None):
    # load torso positions
    info_frames = np.load(os.path.join(path, 'info_frames.npz'))

    # check if limit is set
    joints = info_frames['joints_3d_world']
    limit = min(limit or joints.shape[0], joints.shape[0])

    # load scenes
    scene = np.array([
        cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
        for file in get_jpeg_files(os.path.join(path, 'scene_inputs/'))[:limit]
    ])

    return scene, info_frames

#Prepare input data to NN training
def create_goal_net_dataset(low_limit, secs, fps, scenes, heatmaps):

    inputs = []
    t = secs * fps + 1
    for i in range(0, scenes.shape[0]): #since the number of scenes was already subtracted by low_limit when load_preprocessed()
        inputs.append(
            (scenes[i], heatmaps[i])
        )

    return inputs

#ground truth goal positions
def compute_gt_goals(low_limit, secs, fps, sampling_step, info_frames):
    gt_goals = []
    t = secs * fps
    p_world = info_frames['joints_3d_world'][:,TORSO_IDX]
    world2cam = info_frames['world2cam_trans']
    cam_intrinsics =  info_frames['intrinsics_heatmap']
    num_frames = len(p_world)
    print(num_frames)
    goal_height, goal_width = (256//4,448//4)#GoalNet.heatmap_size

    for i in range(low_limit, num_frames - t -1, sampling_step):
        goal = world2image(p_world[i+t],world2cam[i],cam_intrinsics[i])
        goal = np.clip(goal, (0,0), (goal_width-1, goal_height-1))
        gt_goals.append(goal)

    return gt_goals

def load_test_data(path, index, t):
    inputs = []
    goals = []

    # load torso positions
    info_frames = np.load(os.path.join(path, 'info_frames.npz'))
    joints = info_frames['joints_3d_world']

    p_world = info_frames['joints_3d_world'][:,TORSO_IDX]
    world2cam = info_frames['world2cam_trans']
    cam_intrinsics =  info_frames['intrinsics_heatmap']
    goal_height, goal_width = (256//4,448//4)#GoalNet.heatmap_size

    # load scenes
    scene =  cv.imread(get_jpeg_files(os.path.join(path, 'scene_inputs/'))[index]).swapaxes(0, 2).swapaxes(1, 2)
    print(scene.shape)

    # load heatmaps
    heatmap = cv.imread(get_jpeg_files(os.path.join(path, 'heatmaps/'))[index]).swapaxes(0, 2).swapaxes(1, 2) #np.array([cv.imread(file) for file in get_jpeg_files(os.path.join(path, 'heatmaps/'))[index:index+n]]).swapaxes(1, 3).swapaxes(2,3)
    print("heatmap shape:", np.shape(heatmap))
    #goal_heatmaps = cv.imread(get_jpeg_files(os.path.join(path, 'goal_heatmaps/'))[index]).swapaxes(0, 2).swapaxes(1, 2)
    inputs.append((scene, heatmap))
    goal = world2image(p_world[index+t],world2cam[index],cam_intrinsics[index])
    goal = np.clip(goal, (0,0), (goal_width-1, goal_height-1))
    goals.append(goal)

    return inputs, goals
    
def goal_net_training_data(scene, info_frames, t, output_size, sigma_joints:float=None, sigma_goal:float=None):
    # init parameters
    ## determine number of samples
    s = min(scene.shape[0], info_frames['joints_3d_world'].shape[0]-t)

    ## approximate sigma
    if sigma_joints is None:
        sigma_joints = np.sum(output_size) / 24#16#64

    if sigma_goal is None:
        sigma_goal = np.sum(output_size) / 16

    # create subsets
    ## scene inputs
    scene = scene[:s].swapaxes(3,2).swapaxes(2,1)

    ## load info frames
    p_w = info_frames['joints_3d_world'][:,TORSO_IDX]
    p_j = info_frames['joints_2d_scene']
    t_wc = info_frames['world2cam_trans']
    t_ic = info_frames['intrinsics_scene']
    ### scale intrinsics for goal heatmap size
    scene_height, scene_width = scene.shape[-2:]
    heatmap_height, heatmap_width = output_size
    t_ic[:,0,(0,2)] *= heatmap_width / scene_width
    t_ic[:,1,(1,2)] *= heatmap_height / scene_height

    ## joint inputs
    joints = np.zeros((s, 1, heatmap_height, heatmap_width))
    for i in range(s):
        j = p_j[i]
        j[:,1] *= heatmap_height / scene_height
        j[:,0] *= heatmap_width / scene_width
        # swap x and y
        j = np.flip(j, axis=1)
        joints[i,0] = fill_multi_gaussian(output_size, j, sigma_joints)*128
    # add scene image
    joints[:,0] += np_batch_mean_pool(
        np.mean(scene, axis=1)/2.,
        scene_width // heatmap_width
    )

    ## goals
    ### create goals and heatmaps
    goals = np.zeros((s,2))
    heatmaps = np.zeros((s,1,*output_size), dtype=int)
    for i in range(s):
        goals[i] = np.clip(world2image(p_w[t+i], t_wc[i], t_ic[i]),
            (0,0), (heatmap_width-1, heatmap_height-1))
        heatmaps[i,0] = fill_gaussian(output_size, np.flip(goals[i]), sigma_goal) * 255 + 15


    return scene, joints, goals, heatmaps

