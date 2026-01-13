from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F

from models.extractor import VitExtractor



class LossM(torch.nn.Module):

    def __init__(self, cfg,device):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)


        self.global_transform = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=cfg['lambda_global_ssim'],
        )


    def forward(self, outputs, inputs):
        loss_G = 0
        loss_G = checkpoint(self.calculate_global_ssim_loss, outputs, inputs) *self.lambdas['lambda_global_ssim']
        loss_G += checkpoint(self.calculate_crop_cls_loss, outputs, inputs)*self.lambdas['lambda_global_cls']
        return loss_G

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs): 
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

