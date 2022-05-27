#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:03
# @Author  : ZSH

import torch
class VggConfig():
    def __init__(self):
        self.train_path="./data/train"
        self.test_path="./data/test"

        self.image_size=64
        self.batch_size=64

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

