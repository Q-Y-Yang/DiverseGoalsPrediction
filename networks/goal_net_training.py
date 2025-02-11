import torch
import numpy as np
import os
from tensorboardX import SummaryWriter

def train(model, inputs, goals, gt_goals, batch_size, epochs, shuffle=False):
    writer = SummaryWriter( 'runs')
    history = []
    last_epoch_distance_loss = 40
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}:")
        #writer.add_graph(model, [torch.rand(1,3,256,448), torch.rand(1,3,64,112)])
        model.train(True)
        epoch_loss, epoch_distance_loss = train_one_epoch(model, inputs, goals, gt_goals, batch_size, shuffle=shuffle)
        model.train(False)
        print("average distance loss in an EPOCH:", epoch_distance_loss)
        if epoch_distance_loss < last_epoch_distance_loss:
            print(f"Saving model and metrics at: {'./trained/goal_net/'}")
            # save model
            model.save(os.path.join('./trained/goal_net/', 'epoch' + str(e) + '_loss'+ str(epoch_distance_loss) +'_model.pth'))
            last_epoch_distance_loss = epoch_distance_loss
        # TODO validation
        # print some epoch metrics
        #print(f"\r  Loss: {epoch_loss:>.4f}".ljust(80)[:80])

        history.append(epoch_distance_loss)

        writer.add_scalar('loss', epoch_distance_loss , e)
        
    writer.close()

    return history

def train_one_epoch(model, inputs, goals, gt_goals, batch_size, shuffle=False):

    n = len(goals)
    running_loss = 0.
    sum_distance_loss = 0.
    batches = int(np.ceil(n / batch_size))

    idcs = np.arange(n)
    if shuffle:
        np.random.shuffle(idcs)

    for b in range(0, n, batch_size):
        # select batch
        batch_idcs = idcs[b:b+batch_size]
        batch_inputs = [inputs[i] for i in batch_idcs]
        batch_goals = [goals[i] for i in batch_idcs]
        batch_gt_goals = [gt_goals[i] for i in batch_idcs]
        # restructure
        scene_inputs, heatmap_inputs = zip(*batch_inputs)
        print('scene input shape:', np.shape(scene_inputs))
        # print('lens heatmap input:', len(heatmap_inputs[4]))
        scene_inputs = torch.tensor(np.stack(scene_inputs), dtype=torch.float).to(model.device)
        heatmap_inputs = torch.tensor(np.stack(heatmap_inputs), dtype=torch.float).to(model.device)

        goals_batch = torch.tensor(np.stack(batch_goals), dtype=torch.float).to(model.device)            

        # train
        loss, distance_loss = train_on_batch(model, scene_inputs, heatmap_inputs, goals_batch, batch_gt_goals)
        running_loss += loss
        batch_avg_distance_loss = distance_loss / batch_size
        print("average distance loss in this batch:", batch_avg_distance_loss)
        sum_distance_loss += batch_avg_distance_loss
        #print(f"\r  [{int(b/batch_size)+1}/{batches}] loss: {running_loss / (b/batch_size+1):>.2f} - batch loss: {loss:>.2f}".ljust(80)[:80], end='')

        #x = b
        y = loss
        #writer.add_scalar('batch', x, b)

    return running_loss / batches, sum_distance_loss / batches


def train_on_batch(model, scene_inputs, heatmap_inputs, goals, gt_goals):
    # reset optimizer
    model.optimizer.zero_grad()

    # forward pass
    outputs = model(scene_inputs, heatmap_inputs)

    # compute loss
    loss = model.loss_function(goals, *outputs)   #
    distance_loss = model.distance_loss(outputs[0], gt_goals)
    loss.backward()

    # adjust weights
    model.optimizer.step()

    # return loss
    return loss.item(), distance_loss