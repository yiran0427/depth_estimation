import torch
from resnet import MyNet

class Model(object):
    def __init__(self, model_path):
        self.model = MyNet()
        self.path = model_path

    def train(self):
        pass

    def test(self):
        pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
 
