import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.Sigmoid()
    )


class AlignFusion(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, gate='simple'):  # to do: 增加feature2的channel
        super(AlignFusion, self).__init__()

        self.down_1 = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_2 = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)
        if gate == 'simple':
            self.gate = SimpleGate(inplane, 2)
        elif gate == "":
            self.gate = None
        else:
            raise ValueError("no this type of gate")
    
    def forward(self, feature1, feature2):
        x1 = feature1
        x2 = feature2
        h, w = x1.size()[2:]
        size = (h, w)

        x1 = self.down_1(x1)
        x2 = self.down_2(x2)

        flow = self.flow_make(torch.cat([x1, x2], 1))
        # print("\n\n\n")
        # print(flow.shape)

        # if self.gate:
        #     flow = self.gate(feature1, flow)
        feature1_warp = self.flow_warp(feature1, flow, size=size)
        return feature1 + feature1_warp

    
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class SimpleGate(nn.Module):
    def __init__(self, inplane, outplane):
        super(SimpleGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(inplane, 2, 1),
            nn.Sigmoid()
        )


    def forward(self, feats, x):
        size = feats.size()[2:]
        flow_gate = self.gate(feats)
        x = x*flow_gate
        return x

