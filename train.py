#!/usr/bin/env python3

import os
import argparse

from networks.resnet_masked import ResNetMasked


if __name__ == '__main__':

    model = ResNetMasked(load_weights=False, epochs=100, batch_size=64)

    model.train()
