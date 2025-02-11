from networks import GoalNet
from utils.data import load_preprocessed, create_goal_net_dataset, load_test_data
import torch
import numpy as np
from networks.goal_net import soft_argmax_2d
import matplotlib.pyplot as plt
from utils import nms
import cv2
import time
import argparse

#Constants
N, T = 1, 10
INDEX = 1183
MODEL_PATH = './trained/goal_net/model.pth'
ROI_SIZE = 16
NUM_ROI = 8
DATA_DIR = '/PATH_TO_DATA/preprocessed'

def post_processing(NUM_ROI, heatmap_outputs):
    # Bounding boxes: (start_x, start_y, end_x, end_y)
    bounding_boxes = []
    confidence_score = []
    heatmap_outputs_copy = heatmap_outputs.detach().clone()
    for i in range(NUM_ROI):
        argmax_index = np.unravel_index(heatmap_outputs_copy[0][0].argmax(), heatmap_outputs_copy[0][0].shape)
        max_value = heatmap_outputs_copy[0][0][argmax_index[0]][[argmax_index[1]]].item()
        heatmap_outputs_copy[:, :, argmax_index[0]-int(ROI_SIZE/2):argmax_index[0]+int(ROI_SIZE/2), argmax_index[1]-int(ROI_SIZE/2):argmax_index[1]+int(ROI_SIZE/2)] = heatmap_outputs.min()
        bounding_boxes.append((argmax_index[0] - int(ROI_SIZE/2), argmax_index[1] - int(ROI_SIZE/2), argmax_index[0] + int(ROI_SIZE/2), argmax_index[1] + int(ROI_SIZE/2)))
        if i==0:
            confidence_score.append(max_value)
        else:
            confidence_score.append(max_value/confidence_score[0])

    confidence_score[0] = 1

    heatmap_outputs = heatmap_outputs.squeeze(dim=1).detach().numpy().swapaxes(0, 1).swapaxes(1, 2).astype(int)

    cv2.imwrite("heatmap_output.png", cv2.applyColorMap((heatmap_outputs*20).astype(np.uint8), cv2.COLORMAP_PARULA))
    #heatmap_outputs = np.resize(heatmap_outputs, (64, 112)).astype(np.uint8)
    print("VARIANCE", np.var(heatmap_outputs))
    picked_boxes, picked_score = nms.nms(bounding_boxes, confidence_score, threshold=0.08)

    goals=[]
    for i in range(len(picked_boxes)):
        max_index_in_box = np.unravel_index(heatmap_outputs[picked_boxes[i][0]:picked_boxes[i][2], picked_boxes[i][1]:picked_boxes[i][3]].argmax(), (16,16))
        goals.append((max_index_in_box[0]+picked_boxes[i][0], max_index_in_box[1]+picked_boxes[i][1]))
    
    return picked_boxes, goals, heatmap_outputs


def main(DATA_DIR:str=DATA_DIR, N:int=N, T:int=T, INDEX:int=INDEX,
            ROI_SIZE:int=ROI_SIZE, NUM_ROI:int=NUM_ROI,
            MODEL_PATH:str=MODEL_PATH, *args, **kwargs):
    #load model and weights
    goal_net = GoalNet(N)
    goal_net.load(MODEL_PATH)

    #load test data
    inputs_train, goal_position = load_test_data(DATA_DIR, INDEX, T)

    scene_inputs, heatmap_inputs = zip(*inputs_train)
    scene_inputs = torch.tensor(np.stack(scene_inputs), dtype=torch.float)
    heatmap_inputs = torch.tensor(np.stack(heatmap_inputs), dtype=torch.float)

    goal_net.train(False)
    start_time = time.time()
    heatmap_outputs, mu, log_var = goal_net(scene_inputs, heatmap_inputs)
    end_time = time.time()
    print(f"Average inference time: {(end_time - start_time) * 1000:.2f} ms")
    #decease one dimension for all data, since only one frame is loaded.
    scene_inputs = scene_inputs.detach().numpy()[0].swapaxes(0, 1).swapaxes(1, 2)

    picked_boxes, goals, heatmap_outputs = post_processing(NUM_ROI, heatmap_outputs)

    #plotting
    fig, axs = plt.subplots(1, 2)
    scene_inputs = cv2.resize(scene_inputs, (112,64))
    add = np.zeros_like(scene_inputs)
    add[:,:,0:1] = heatmap_outputs

    scene_heatmap = cv2.addWeighted(scene_inputs, 0.9, add*200, 0.1, 0)
    scene_heatmap = scene_heatmap.astype(int)
    cv2.imwrite("prediction_output.png", cv2.cvtColor(scene_heatmap.astype(np.float32), cv2.COLOR_RGB2BGR))
    axs[1].imshow(scene_heatmap)
    heatmap_outputs = np.resize(heatmap_outputs, (64, 112))
    axs[0].imshow(heatmap_outputs)
    axs[1].scatter(x=goal_position[0][0], y=goal_position[0][1], c='g', s=1)
    goal_position[0][0] = np.clip(goal_position[0][0], 1, 112)
    goal_position[0][1] = np.clip(goal_position[0][1], 1, 64)

    for i in range(len(picked_boxes)):
        rect = plt.Rectangle((picked_boxes[i][1], picked_boxes[i][0]), ROI_SIZE, ROI_SIZE, fill=False, edgecolor='red',linewidth=1)
        #axs[0].add_patch(rect)
        axs[1].add_patch(rect)
        axs[1].scatter(x=goals[i][1], y=goals[i][0], c='r', s=1)
        distance_loss = np.linalg.norm(goal_position[0] - np.array((goals[i][1], goals[i][0])))
        print("distance loss:", distance_loss)
    plt.show()

# Entry point
if __name__ == '__main__':
    # fetch command line args
    parser = argparse.ArgumentParser()

    # model parameter
    parser.add_argument('-n', type=int, default=N)
    parser.add_argument('-t', type=int, default=T)
    
    # data parameter
    parser.add_argument('-d', '--data', type=str, default=DATA_DIR)
    parser.add_argument('-i', type=int, default=INDEX)

    # Model IO
    parser.add_argument('-c', '--checkpoint', type=str, default=MODEL_PATH)

    # post-processing parameter
    parser.add_argument('-r', '--num_roi', type=int, default=NUM_ROI)
    parser.add_argument('-s', '--roi_size', type=int, default=ROI_SIZE)

    # run program
    main(**vars(parser.parse_args()))


