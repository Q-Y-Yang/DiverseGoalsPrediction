# Imports
import argparse
import json
import logging
import os
import torch
import numpy as np
from networks import GoalNet, train_goal_net
from utils.data import (
    get_preprocessed_subdirectories,
    load_preprocessed,
    goal_net_training_data
)

# Constants
T = 10 # 2 seconds goal (5 FPS)
Z = 200  # latent space size
SIGMA_GOAL = None # for output heatmaps
SIGMA_JOINTS = None # for output heatmaps

# Default setting
## IO
DATA_DIR = 'PATH_TO_DATA/'
LIMIT = None
CHECKPOINT = None#'./trained/goal_net/model.pth'
OUT_DIR = './trained/goal_net/'

## Training
EPOCHS = 2
BATCH_SIZE = 64
LOSSES = 'mse'

## Hardware
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging
LOG_LEVEL = logging.INFO

# Program
def main(data:str=DATA_DIR, limit:int=LIMIT, t:int=T, z:int=Z,
            sigma_goal:float=SIGMA_GOAL, sigma_joints:float=SIGMA_JOINTS,
            epochs:int=EPOCHS, batch_size:int=BATCH_SIZE, losses:str=LOSSES, 
            output:str=OUT_DIR, checkpoint:str=CHECKPOINT, device:str=DEVICE, 
            log_level:int=LOG_LEVEL, *args, **kwargs):
    
    # set loglevel
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s - %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # prepare output directory
    logging.info(f"Preparing output direcotry {output}")
    os.makedirs(output, exist_ok=True) # ensure output dir exists

    # load data
    scenes, joints, goals, heatmaps = [], [], [], []
    for d in get_preprocessed_subdirectories(data):
        logging.info(f"Loading data from {d}")
        scene, info_frames = load_preprocessed(d, limit)

        logging.info(f"Createing training set for GoalNet")
        s, j, g, h = goal_net_training_data(
            scene, info_frames, t, GoalNet.output_size, sigma_joints, sigma_goal)
        
        scenes.append(s)
        joints.append(j)
        goals.append(g)
        heatmaps.append(h)

        del s, j, g, h
        torch.cuda.empty_cache()
    
    logging.info(f"Joining data into one dataset")
    scenes = np.concatenate(scenes)
    joints = np.concatenate(joints)
    goals = np.concatenate(goals)
    heatmaps = np.concatenate(heatmaps)

    del info_frames
    torch.cuda.empty_cache()

    # init goal-net
    logging.info(f"Initializing new GoalNet model")
    goal_net = GoalNet(z).to(device)
    if checkpoint:
        logging.info(f"Importing weights from {checkpoint}")
        goal_net.load(checkpoint)

    # train
    success, history = train_goal_net(
        model=goal_net,
        inputs=scenes,
        conditions=joints,
        outputs=heatmaps,
        targets=goals,
        batch_size=batch_size,
        epochs=epochs,
        losses=losses.split(','),
        shuffle=True
    )

    if not success:
        logging.warning(f"Aborting training")
        return -1

    logging.info(f"Saving model and metrics at: {output}")
    # save model
    goal_net.save(os.path.join(output, 'model.pth'))

    # write history
    with open(os.path.join(output, 'training_history.json'), mode='w+') as file:
        json.dump(history, file)


# Entry point
if __name__ == '__main__':
    # fetch command line args
    parser = argparse.ArgumentParser()

    # model parameter
    #parser.add_argument('-n', type=int, default=N)
    parser.add_argument('-t', type=int, default=T)
    parser.add_argument('-z', type=int, default=Z)
    parser.add_argument('-sg', '--sigma-goal', type=float, default=SIGMA_GOAL)
    parser.add_argument('-sj', '--sigma-joints', type=float, default=SIGMA_JOINTS)
    
    # data parameter
    parser.add_argument('-d', '--data', type=str, default=DATA_DIR)
    parser.add_argument('-l', '--limit', type=int, default=LIMIT)

    # Model IO
    parser.add_argument('-c', '--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('-o', '--output', type=str, default=OUT_DIR)

    # training parameter
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('-ls', '--losses', type=str, default=LOSSES)

    # hardware parameter
    parser.add_argument('--device', type=str, default=DEVICE)

    # logging parameter
    parser.add_argument('-ll', '--log-level', type=int, default=LOG_LEVEL)

    # run program
    main(**vars(parser.parse_args()))
