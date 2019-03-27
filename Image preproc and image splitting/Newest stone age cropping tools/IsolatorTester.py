# -*- coding: utf-8 -*-
"""
Place this in the folder below test images, and run it to test the performance of the cluster isolator
@author: Jurriez
"""

import numpy as np
import ClusterCropV3a as CC
import os

import cProfile
import pstats

i = 10

TargetImgW = 500
TargetImgH = 500


dirname = os.path.dirname(__file__)
ImagePath = dirname + "/train/"

for j in range(0,i):
    rand = np.round(14990*np.random.uniform(0,1)).astype(np.int)
    rand = rand.astype(str);
    
    zerosstring = "00000000000"
    
    numelN = len(rand)
    AddZeros = 5 - numelN;
    
    numberstring = zerosstring[0:AddZeros] + rand;
    imgname = "img0" + numberstring + ".jpg"
    imgname = ImagePath + imgname
    
    #CC.PictureSnipSnipper(imgname, TargetImgW,TargetImgH)
    cProfile.run('CC.PictureSnipSnipper(imgname, TargetImgW,TargetImgH)', 'myFunction.profile')
    import pstats
    stats = pstats.Stats('myFunction.profile')
    #stats.strip_dirs().sort_stats('time').print_stats()




##images to analyse in a single run. replacing it each run was tiresome.
#images = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
#LazyOverride = 5; # override the loop without having to change the array of namesa
#
#Nloops = np.min([len(images), LazyOverride])
#
#AspectRatio = TargetImgW / TargetImgH
# 
#
#for i in range(0,Nloops):
#    ClusterCrop(images[i])