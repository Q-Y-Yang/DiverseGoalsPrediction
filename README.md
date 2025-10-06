# Diverse Human Movement Goals Prediction
This is the repository for the paper "**Scene-aware Prediction of Diverse Human Movement Goals:**" published on the 5th International Conference on Robotics, Computer Vision and Intelligent Systems (ROBOVIS 2025) .

[Paper Link](https://link.springer.com/chapter/10.1007/978-3-032-00986-9_21)


## Main Dependencies
 * [Python](https://python.org) (Version 3.10)
 * [PyTorch](https://pytorch.org) (Version 2.0)


## Dataset Preprocessing
 * Downscaling the scene images and calculating joint positions for [GTA-IM Dataset](https://github.com/ZheC/GTA-IM-Dataset):

 ```
python preprocssing_gta.py
 ```
 

 * [PROX](https://prox.is.tue.mpg.de/):

 ```
 python preprocessing-prox.py
 ```

 `/utils/data.py`: Contains functions to manipulate raw image data into data usable for training and testing.

 ## Training

 `networks/goal_net.py`: Defines the GoalNet. GoalNet is a variational autoencoder that processes an input images along with a stack of heatmaps into the goal heatmap. The stack of heatmaps contains the joint positions of the human in $N$ previous frames. The output goal heatmap shows the torso position $T$ frames i the future.

`train_goal_net.py`: Contains the code train the GoalNet. Usage:

```bash
python train_goal_net.py -n 1 -t 10 -z 200 -d path/to/data/ -l 100 -c load/model/checkpoint/ -o model/output -e 8 -bs 128 --device cpu
```

Arguments:
* `-n`: Number of context frames
* `-t`: frames in future for goal prediction
* `-z`: latent space size (default 200)
* `-d` | `--data`: path to processed dataset
* `-l` | `--limit`: maximum number of data points (useful for testing on machines with small memory) (default -1 - use all data)
* `-c` | `--checkpoint`: path to a stored model checkpoint to continue training from (default: None)
* `-o` | `--output`: path to store the trained model (default [./trained/goal_net](./trained/goal_net/))
* `-e` | `--epochs`: number of training epochs (default 2)
* `-bs` | `--batch-size`: training batch size (default 64)
* `--device`: specific CUDA device. By default, GPU is preferred over CPU

## Inference
`inference.py`: intializing the network, loading trained model weights, loading a test data, inferecing, and visulizing prediction.

```
python inference.py -n 1 -t 10 -d path/to/data/ -i 300 -c load/model/checkpoint/ -r 8 -s 16
```

Arguments:
* `-n`: Number of context frames, which is 1 in this method.
* `-t`: frames in future for goal prediction.
* `-i`: the index of a test frame among the test set.
* `-d` | `--data`: path to processed test dataset
* `-c` | `--checkpoint`: path to a stored model
* `-r` | `--num_roi`: the number of Region of Interest (ROI) that will be extracted from the predicted goal heatmap.
* `-s` | `--roi_size`: the size in pixel of a Region of Interest (ROI), which depends on the size of goal area set during training.


## Evaluation
`evaluation.py`: extracting possible goal areas with confidence scores from one ouput of the model (though our model can generate various predictions), quantitatively evaluating the Final Displacement Errors (FDEs), and saving the FDEs in a csv file.

Arguments:
* `-n`: Number of context frames, which is 1 in this method.
* `-t`: frames in future for goal prediction.
*  `--dataset`: path to processed test dataset
* `--model_path`: path to a stored model
* `-r` | `--num_roi`: the number of Region of Interest (ROI) that will be extracted from the predicted goal heatmap.
* `-s` | `--roi_size`: the size in pixel of a Region of Interest (ROI), which depends on the size of goal area set during training.
* `--data_length`: the number of test data (useful for testing on machines with small memory)


## Citation
If you use the code in this repository or derive ideas from this work, please cite the following paper:
```
@InProceedings{MultiGoalsPredict,
author="Yang, Qiaoyue
and Weber, Amadeus
and Jung, Magnus
and AI-Hamadi, Ayoub
and Wachsmuth, Sven",
editor="R{\"o}ning, Juha
and Filipe, Joaquim",
title="Scene-Aware Prediction of Diverse Human Movement Goals",
booktitle="Robotics, Computer Vision and Intelligent Systems",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="312--327",
isbn="978-3-032-00986-9"
}
```
