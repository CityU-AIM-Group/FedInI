import torch
import torch.nn.functional as F
from torch import nn


class AttentionModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AttentionModule, self).__init__()

        self.cfg = cfg
        self.fc_cat = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc_a = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc_b = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc_weight = nn.Conv2d(in_channels // 4, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc_weight_sigmoid = nn.Sigmoid()

        torch.nn.init.normal_(self.fc_cat.weight, std=0.01)
        torch.nn.init.constant_(self.fc_cat.bias, 0)
        torch.nn.init.normal_(self.fc_a.weight, std=0.01)
        torch.nn.init.constant_(self.fc_a.bias, 0)
        torch.nn.init.normal_(self.fc_b.weight, std=0.01)
        torch.nn.init.constant_(self.fc_b.bias, 0)
        torch.nn.init.normal_(self.fc_weight.weight, std=0.01)
        torch.nn.init.constant_(self.fc_weight.bias, 0)

    def forward(self, x_part1, x_part2):
        
        fc_a = F.relu(self.fc_a(x_part1), inplace=True)
        fc_b = F.relu(self.fc_b(x_part2), inplace=True)

        if self.cfg.SOLVER.ATT_METHOD == "add":
            fused = fc_a + fc_b
        elif self.cfg.SOLVER.ATT_METHOD == "mul":
            fused = fc_a * fc_b
        elif self.cfg.SOLVER.ATT_METHOD == "cat":
            fused = torch.cat((fc_a, fc_b), 1)
            fused = self.fc_cat(fused)
        else:
            raise NotImplementedError

        fc_weight = self.fc_weight(fused)
        weight_map = self.fc_weight_sigmoid(fc_weight)

        return weight_map


def build_attention(cfg, in_channels):
    return AttentionModule(cfg, in_channels)
