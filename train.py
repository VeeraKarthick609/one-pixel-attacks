#!/usr/bin/env python3

import os
import argparse

from networks.resnet import ResNet


if __name__ == '__main__':

    model = ResNet(load_weights=False, epochs=1)

    model.train()
