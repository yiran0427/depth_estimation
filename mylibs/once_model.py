import os
import torch
import datetime
import collections
import numpy as np

from mylibs.resnet import MyNet
from mylibs.loss import Loss

class Once_Model(object):
    def __init__(self, train_loader, test_loader, device='cpu', current_model=None,
        img_height=256, img_width=512, model_path='output_model', disp_path='output_disp'):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.img_height = img_height
        self.img_width = img_width
        self.model_path = model_path
        self.disp_path = disp_path
        
        if(current_model == None):
            self.model = MyNet().to(self.device)
        else:
            self.model = current_model.to(self.device)
        self.loss_function = Loss(n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self):
        self.model.train()
        start_time = datetime.datetime.now()
        train_loss = []
        
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pt"))
        loss = 0
        for data in self.train_loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']

            self.optimizer.zero_grad()
            disps = self.model(left)
            batch_loss = self.loss_function(disps, [left, right])
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item()

        loss = loss / len(self.train_loader.dataset)
        train_loss.append(loss)

        current_time = datetime.datetime.now()
        print('--- Epoch \tAverage Loss: {:.2f}\tTime: {}'.format(loss, str(current_time - start_time)))

        #self.save(os.path.join(self.model_path, "model.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pt"))

        return train_loss


    def test(self, path):
        self.load(path)
        self.model.eval()

        disparities = np.zeros((len(self.test_loader.dataset), self.img_height, self.img_width), dtype=np.float32)
        disparities_pp = np.zeros((len(self.test_loader.dataset), self.img_height, self.img_width), dtype=np.float32)

        with torch.no_grad():
            for data in self.test_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                disps = self.model(left) 
                disp = disps[0] # [batch, 2, width, height]
                for i in range(len(disp)):
                    disparities[i] = disp[i, 0, :, :].squeeze().cpu().numpy()
                    disparities_pp[i] = post_process_disparity(disp[i, :, :, :].cpu().numpy())

        np.save(self.disp_path + '/disparities.npy', disparities)
        np.save(self.disp_path + '/disparities_pp.npy', disparities_pp)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


# https://github.com/mrharicot/monodepth
def post_process_disparity(disp):
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
    elif isinstance(input, collections.Mapping):
        dic = {}
        for k, v in input.items():
            dic[k] = to_device(v, device=device)
        return dic
    else:
        print('TYPE ERROR!')

