#!/usr/bin/env python3

import os
import argparse

from networks.resnet import ResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



if __name__ == '__main__':

    model = ResNet(load_weights=False, epochs=100, batch_size=64)

    model.train()
