import torch
import os
from functools import partial
from piq import psnr, ssim, gmsd
import numpy as np

class MetricEvaluator:
    def __init__(self, path):
        """ create a metric evaluator for IRFPN denoising """
        self.filename = os.path.join(path, "scores.txt")
        self.metrics = "psnr, ssim, gmsd"
        self.metricFun = [
            partial(psnr, reduction='none'),
            partial(ssim, reduction='none'),
            partial(gmsd, reduction='none')
        ]

        # check the score file and delete if exist
        if os.path.isfile(self.filename):
            os.remove(self.filename)
            
    # set the current header
    def start(self, modelName):
        self.header = modelName + "," + self.metrics
        self.scores = np.zeros((0, len(self.metricFun)))
    
    def evaluate(self, ref, irc):
        """" calculate the statistics of each element """
        # REF, IRC:  BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        ref = ref.view(-1, ref.shape[2], ref.shape[3], ref.shape[4])
        irc = irc.view(-1, irc.shape[2], irc.shape[3], irc.shape[4])

        # get the scores
        mcount = len(self.metricFun)
        score = np.zeros((ref.shape[0], mcount))
        for m in range(mcount):
            score[:, m] = self.metricFun[m](ref, irc)
        
        # append the array
        self.scores = np.append(self.scores, score, axis=0)
    
    # save the resulting scores array to disk
    def save_metrics(self, reduction: str):
        # open the so that we can append multiple results
        with open(self.filename, "a") as log_file:
            # print each score
            if reduction == "none":
                np.savetxt(log_file, self.scores, fmt='%.4f', delimiter=', ', header=self.header)
            elif reduction == "mean":
                np.savetxt(log_file, np.mean(self.scores, axis=0, keepdims=True), fmt='%.4f', delimiter=', ', header=self.header)
            else:
                print("unknown reduction method")
        
        
        