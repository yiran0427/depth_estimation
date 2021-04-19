import torch
import datetime
import collections
import numpy as np

from resnet import MyNet
from loss import MonodepthLoss

class Model(object):
    def __init__(self, device='cpu', epochs=10, save_per_epoch='none', img_height=256, img_width=512, model_path='output_model', disp_path='output_disp'):
        self.device = device
        self.epochs = epochs
        self.save_per_epoch = save_per_epoch
        self.img_height = img_height
        self.img_width = img_width
        self.model_path = model_path
        self.disp_path = disp_path

        self.model = MyNet().to(self.device)
        self.loss_function = MonodepthLoss(n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, dataloader):
        self.model.train()
        start_time = datetime.datetime.now()
        train_loss = []

        for epoch in range(self.epochs):
            loss = 0
            for data in dataloader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                self.optimizer.zero_grad()
                disps = self.model(left)
                batch_loss = self.loss_function(disps, [left, right])
                batch_loss.backward()
                self.optimizer.step()
                loss += batch_loss.item()

            loss = loss / len(dataloader)
            train_loss.append(loss)

            if epoch % 10 == 0:
                current_time = datetime.datetime.now()
                print('--- Epoch {}\tAverage Loss: {:.2f}\tTime: {}'.format(epoch, loss, str(current_time - start_time)))

            if self.save_per_epoch != "none" and epoch % self.save_per_epoch == 0:
                self.save(os.path.join(self.model_path, str(epoch) + ".pt"))

        return train_loss


    def test(self, dataloader):
        self.model.eval()

        disparities = np.zeros((len(dataloader), self.img_height, self.img_width), dtype=np.float32)
        disparities_pp = np.zeros((len(dataloader), self.img_height, self.img_width), dtype=np.float32)

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                data = to_device(data, self.device)
                left = data['left_image']
                print('left ', left.shape)
                disps = self.model(left)
                print('disps ', disps[0].shape)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[idx] = disp[0].squeeze().cpu().numpy()
                disparities_pp[idx] = post_process_disparity(disps[0][:, 0, :, :].cpu().numpy())

        np.save(self.disp_path + '/disparities.npy', disparities)
        np.save(self.disp_path + '/disparities_pp.npy', disparities_pp)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def post_process_disparity(disp):
    print(disp.shape)
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")
