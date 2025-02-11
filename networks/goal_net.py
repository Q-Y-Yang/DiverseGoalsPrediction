import torch
import cv2 as cv
import numpy as np

# Functions
def soft_argmax_2d(inputs):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W)
	Return: 2D coordinates in shape (batch_size, channel, 2)
	"""
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W = inputs.shape
	soft_max = torch.nn.functional.softmax(inputs.reshape(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(inputs.shape)
	indices_kernel = torch.arange(start=0,end=H*W, device=soft_max.device).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2)
	x = indices%W
	y = (indices/W).floor()%H
	coords = torch.stack([x,y],dim=2)

	return coords


# Classes
## Submodules
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_chanels, in_stride, out_stride, *args, **kwargs) -> None:
        super(ConvBlock, self).__init__(*args, **kwargs)

        # layers
        self.in_conv = torch.nn.Conv2d(in_channels, out_chanels, 3, in_stride, 1)
        self.in_batch_norm = torch.nn.BatchNorm2d(out_chanels)
        self.in_relu = torch.nn.ReLU()

        self.out_conv = torch.nn.Conv2d(out_chanels, out_chanels, 3, out_stride, 1)
        self.out_batch_norm = torch.nn.BatchNorm2d(out_chanels)
        self.out_relu = torch.nn.ReLU()

    def forward(self, x):
        # in convolution
        y = self.in_conv(x)
        y = self.in_batch_norm(y)
        y = self.in_relu(y)

        #out convolution
        y = self.out_conv(y)
        y = self.out_batch_norm(y)
        y = self.out_relu(y)

        # return result
        return y


class GoalNet(torch.nn.Module):
    scene_size = (256,448)
    input_chanels = 3
    heatmap_size = tuple(x//4 for x in scene_size)
    def __init__(self, n_input_time_frames:int, latent_dim:int=200, *args, **kwargs) -> None:
        super(GoalNet, self).__init__(*args, **kwargs)
        scene_size = (256,448)
        input_chanels = 3
        self.heatmap_size = tuple(x//4 for x in scene_size)
        # attributes
        self._latent_dim = latent_dim
        self._n_input_time_frames = n_input_time_frames

        self._device_param = torch.nn.Parameter(torch.empty(0))

        # input processing
        self.scene_input_featurs = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_chanels, 64, 7, 2, 3),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(64, 64, 1, 1)
        )
        self.heatmaps_features = ConvBlock(self._n_input_time_frames * self.input_chanels, 64, 1, 1)

        # encoder
        self.encoder_featrues = torch.nn.Sequential(
            ConvBlock(128, 128, 2, 1),
            ConvBlock(128, 256, 2, 1),
            ConvBlock(256, 512, 2, 1),
            torch.nn.AvgPool2d((8, 14)),
            torch.nn.Flatten()
        )

        # latent space distribution
        self.latent_mu = torch.nn.Linear(512, latent_dim)
        self.latent_ln_var = torch.nn.Linear(512, latent_dim)

        # image rescale
        desired_size = (8, 14)
        factor = tuple(x//y for x, y in zip(self.heatmap_size, desired_size))#self.scene_size
        self.input_to_latent_size = torch.nn.AvgPool2d(factor, factor)

        # decoder
        self.decoder_features = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim + self.input_chanels * self._n_input_time_frames, 512, 3, 1, 1),#
            ConvBlock(512, 512, 1, 1),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(512, 256, 1, 1),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(256, 128, 1, 1),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(128, 64, 1, 1)
        )
        self.pred_heatmap = torch.nn.Conv2d(64, 1, 1, 1, 0)

        # loss fucntions
        self.l1_loss = torch.nn.L1Loss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    @property
    def device(self):
        return self._device_param.device

    def encode(self, scene_input, heatmaps):
        # compute scene features
        x_scene = self.scene_input_featurs(scene_input)
        # compute heatmap features
        x_heatmaps = self.heatmaps_features(heatmaps)
        # combine scene and heatmaps features
        x = torch.concatenate((x_scene, x_heatmaps), dim=1)
        # compute encoder features
        x = self.encoder_featrues(x)
        # compute mean and std
        return self.latent_mu(x), self.latent_ln_var(x)
    
    def reparameterize(self, mu, ln_var, temperature):
        # reshape for sampling
        sigma = torch.reshape(ln_var, shape=(*ln_var.shape, 1, 1))
        mu = torch.reshape(mu, shape=(*mu.shape, 1, 1))
        print("mu shape:", mu.size())
        # sample from latent space using mean and stdd
        epsilon = torch.randn(size=(sigma.shape[0], self._latent_dim, 8, 14)).to(sigma.device)
        print("epsilin_size:", epsilon.size())
        return epsilon * sigma * temperature + mu

    def decode(self, z, img):
        # condition on input scled down image
        c = self.input_to_latent_size(img)
        #print("c size:", c.size())
        y = torch.cat((z, c), dim=1)
        print("size after cat:", y.size())
        # compute decoder features from latent space sample
        y = self.decoder_features(y)
        # predict heatmap
        return self.pred_heatmap(y)


    def forward(self, scene_input, heatmaps, temperature=1):
        # compute scene features
        mu, ln_var = self.encode(scene_input, heatmaps)
        # sample from latent space
        z = self.reparameterize(mu, ln_var, temperature)
        print("z size:", z.size())
        # decode latent output
        pred_heatmap = self.decode(z, heatmaps)

        return pred_heatmap, mu, ln_var
    
    def loss_function(self, true_result, pred_result, mu, ln_var):
        #pred_goal = soft_argmax_2d(pred_result)
        #pred_loss = self.l1_loss(true_result.unsqueeze(1), pred_goal)
        loss_func = torch.nn.MSELoss()

        pred_loss = loss_func(pred_result, true_result)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + ln_var - mu ** 2 - ln_var.exp(), dim = 1), dim = 0)
        print('pred_loss:', pred_loss)
        print('kld_loss:', kld_loss)
        return pred_loss + kld_loss
    
    def distance_loss(self, pred_result, gt_goals):
        argmax = soft_argmax_2d(pred_result).squeeze(dim=1).cpu().detach().numpy()
        #limit the goal position within the image
        gt_goals[0] = np.clip(gt_goals[0], 1, self.heatmap_size[1])
        gt_goals[1] = np.clip(gt_goals[1], 1, self.heatmap_size[0])
        distance_loss = np.linalg.norm(np.linalg.norm(gt_goals - argmax, axis=1), ord=1)

        return distance_loss
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))


