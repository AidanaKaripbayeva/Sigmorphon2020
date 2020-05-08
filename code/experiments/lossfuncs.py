from data import feature_converters, SigmorphonData_Factory
from data.sigmorphon2020_data_reader import *
import logging
from models import ModelFactory
from optimizers import OptimizerFactory
import os
import pickle
import shutil
import time
import torch
import wandb

class AutoPaddedLoss(object):
    def __init__(self, loss, pad_index=Alphabet.stop_integer, device=None):
        self.loss = loss
        self.pad_index = pad_index
        self.device = device
    
    def __call__(self, probs, target):
        L_probs = probs.shape[0]
        L_target = target.shape[0]
        
        
        probs = probs.to(self.device)
        target = target.to(self.device)
        
        if L_probs < L_target:
            #TODO: Fewer assumptions about the shape of probs
            #extra_probs = torch.zeros(L_target-L_probs, probs.shape[1],device=self.device)
            #extra_probs[:,self.pad_index] = 1
            #probs = torch.cat([probs, extra_probs])
            
            #This version should ensure that the last probs contribute to the gradient
            probs = torch.cat([probs] + [probs[-1:]]*(L_target-L_probs))
            
        
        if L_target < L_probs:
            target = torch.cat([target] + [torch.full((L_probs-L_target,),self.pad_index,dtype=torch.long,device=self.device)])
        
        return self.loss(probs,target)
    
    def to(self, target_device):
        self.device = target_device
        self.loss = self.loss.to(self.device)
        return self


__all__ = ["AutoPaddedLoss"]
