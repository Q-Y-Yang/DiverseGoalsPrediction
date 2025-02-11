from networks import GoalNet
from utils.data import load_preprocessed, create_goal_net_dataset, compute_gt_goals
import torch
import numpy as np
from utils import nms
import cv2
import os
import argparse
import csv

def multi_goals(heatmap_outputs, NUM_ROI, ROI_SIZE):
    # Bounding boxes: (start_x, start_y, end_x, end_y)
    heatmap_outputs_copy = heatmap_outputs.detach().clone()
    data_size = heatmap_outputs_copy.size()[0]
    bounding_boxes = [[] for d in range(data_size)]
    confidence_scores = [[] for d in range(data_size)]

    for h in range(data_size):
        for r in range(NUM_ROI):
            argmax_index = np.unravel_index(heatmap_outputs_copy[h][0].argmax(), heatmap_outputs_copy[h][0].shape)
            max_value = heatmap_outputs_copy[h][0][argmax_index[0]][[argmax_index[1]]].item()
            heatmap_outputs_copy[h, :, argmax_index[0]-int(ROI_SIZE/2):argmax_index[0]+int(ROI_SIZE/2), argmax_index[1]-int(ROI_SIZE/2):argmax_index[1]+int(ROI_SIZE/2)] = heatmap_outputs.min()
            bounding_boxes[h].append((argmax_index[0] - int(ROI_SIZE/2), argmax_index[1] - int(ROI_SIZE/2), argmax_index[0] + int(ROI_SIZE/2), argmax_index[1] + int(ROI_SIZE/2)))
            if r==0:
                confidence_scores[h].append(max_value)
            else:
                confidence_scores[h].append(max_value/confidence_scores[h][0])
        confidence_scores[h][0] = 1

    return bounding_boxes, confidence_scores


def eval_multi_goals(heatmap_outputs, bounding_boxes, confidence_scores, gt_goals):
    picked_boxes = []
    picked_scores = []
    errors = []
    data_size = len(bounding_boxes)

    goals=[[] for d in range(data_size)]
    for b in range(data_size):
        picked_box, picked_score = nms.nms(bounding_boxes[b], confidence_scores[b], threshold=0.08)
        for i in range(len(picked_box)):
            max_index_in_box = np.unravel_index(heatmap_outputs[b, :, picked_box[i][0]:picked_box[i][2], picked_box[i][1]:picked_box[i][3]].argmax(), (16,16))
            goals[b].append((max_index_in_box[1]+picked_box[i][1], max_index_in_box[0]+picked_box[i][0]))
        picked_boxes.append(picked_box)
        picked_scores.append(picked_score)
        errors.append(min([np.linalg.norm(goals[b][i] - gt_goals[b]) for i in range(len(goals[b]))]))
        # print(f"FDEs:", {[np.linalg.norm(goals[b][i] - gt_goals[b]) for i in range(len(goals[b]))]})
        # print(f"Minimal FDE:", {min([np.linalg.norm(goals[b][i] - gt_goals[b]) for i in range(len(goals[b]))])})

    return errors

def evalution(path2data, N, T, model_path, start_frame, data_length, num_roi, roi_size):

    goal_net = GoalNet(N)
    goal_net.load(model_path)
    info_frames = np.load(os.path.join(path2data, 'info_frames.npz'))
    scenes, heatmaps, _ = load_preprocessed(path2data, start_frame, start_frame + data_length, 1)
    inputs_train = create_goal_net_dataset(start_frame, 2, 5, scenes, heatmaps)
    gt_goals = compute_gt_goals(start_frame, 2, 5, 1, info_frames)

    scene_inputs, heatmap_inputs = zip(*inputs_train)
    scene_inputs = torch.tensor(np.stack(scene_inputs), dtype=torch.float)
    heatmap_inputs = torch.tensor(np.stack(heatmap_inputs), dtype=torch.float)

    goal_net.train(False)
    heatmap_outputs, mu, log_var = goal_net(scene_inputs, heatmap_inputs)

    #single goal error
    #avg_error = goal_net.distance_loss(heatmap_outputs, gt_goals)

    bbx, confidence_scores = multi_goals(heatmap_outputs, num_roi, roi_size)
    avg_error = eval_multi_goals(heatmap_outputs, bbx, confidence_scores,gt_goals)
    return avg_error

def main(args):
    path2data = args.dataset
    N = args.N
    T = args.T
    model_path = args.model_path
    num_roi = args.num_roi
    roi_size = args.roi_size
    data_length = args.data_length
    start_frame = 100

    evaluation_result = evalution(path2data, N, T, model_path, start_frame, data_length, num_roi, roi_size)

    with open('evaluation.csv','a+',newline='\r\n') as f:
        csv_write = csv.writer(f)
        data_row = evaluation_result
        csv_write.writerow(data_row)

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--N', type=int, default=1)
    parser.add_argument('-t', '--T', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='/PATH_TO_DATA/preprocessed')
    parser.add_argument('--model_path', type=str, default='./trained/goal_net/model.pth')
    parser.add_argument('--num_roi', type=int, default=20)
    parser.add_argument('--roi_size', type=int, default=12)
    parser.add_argument('--data_length', type=int, default=100)
    main(parser.parse_args())
    
