#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:49:03 2020

@author: sayan

Modified: the body was hardcoded to `.to("cpu")`, which pinned the whole package
to the CPU. It now targets the device resolved by scripts/device.py, which
defaults to CPU so existing behaviour is unchanged unless $FMBENCH_DEVICE is set.
"""
from scripts.device import get_device


def gpu(x):
    return x.to(get_device())
