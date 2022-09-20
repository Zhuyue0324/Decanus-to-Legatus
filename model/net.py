from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Lifter(nn.Module):
    def __init__(self, num_keypoints=17, terms=3):
        super(Lifter, self).__init__()
        self.upscale = nn.Linear(num_keypoints*terms-terms, 1024)
        self.res_1 = res_block()
        self.res_2 = res_block()
        self.res_3 = res_block()
        self.pose3d = nn.Linear(1024, num_keypoints*3-3)

    def forward(self, p2d):
        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_1(x))
        x = nn.LeakyReLU()(self.res_2(x))
        x = nn.LeakyReLU()(self.res_3(x))
        x = self.pose3d(x)
        return x


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x0 = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += x0

        return x
