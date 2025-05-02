import sys
import torch
import torch.nn as nn
import os
import logging

def get_model(arch, num_classes=2, fp16=False, multi_channels=False):
    print(multi_channels)
    if multi_channels:
        import nets.resnet_4_channels as resnet
    else:
        import nets.resnet as resnet
    if arch.startswith('resnet'):
        model = resnet.__dict__[arch](num_classes=num_classes, fp16=fp16)
    else:
        logging.info('arch not supported.')
        sys.exit(-1)
    return model

