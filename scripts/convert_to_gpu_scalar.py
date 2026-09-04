# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:08:35 2020

@author: sayan

Modified: see convert_to_gpu.py. Device now comes from scripts/device.py.
"""
import torch

from scripts.device import get_device


def gpu_ts(x):
    x = torch.tensor(x).float()
    return x.to(get_device())
