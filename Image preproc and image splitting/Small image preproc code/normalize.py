# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import tensorflow as tf
import numpy as np
import cv2 


def NormalizeMatrix(Matrix, rescale):
    "Normalize a matrix between 0 and rescale value"
    Max = np.max(Matrix)
    Matrix = np.round((Matrix/Max)*rescale)
    
    return Matrix

def PlayWithImage( img ):
    
    #Create a copy of the image after reading it to compare later.
    imgo = cv2.imread(img,0)
    img = imgo

    # Firstly normalize the image to make it easier to handle very dark photos in particular.
    img = NormalizeMatrix(img, MaxGrayScale)
    
    # Figure out how many cells the current resolution can support, currently not compensating for irregular sectionsizes
    CellsW = np.int(img.shape[1]/SectionSize)
    CellsH = np.int(img.shape[0]/SectionSize)
    TCells = CellsW*CellsH
    
    # Pre-allocate some memory space
    CellInfo = np.zeros((3,np.int(TCells)))
    CellMap = np.zeros((CellsH,CellsW))    
    BinarySelectionMap = np.zeros(img.shape)
    
    # cc is a simple counter but it is also used as a Cell ID, to couple it to its origin x,y coordinates in CellInfo
    cc = 0
    for i in range(0,CellsW):
        for j in range(0,CellsH):
            Cellsum = 0
            Cell = img[j*SectionSize: (j+1)*SectionSize -1, i*SectionSize: (i+1)* SectionSize -1]
            BinarySelection = np.zeros(Cell.shape)*0
            BinarySelection[Cell>Threshold] = 1
            Cellsum = np.sum(BinarySelection)
            
            CellInfo[0,cc] = i*SectionSize
            CellInfo[1,cc] = j*SectionSize
            CellInfo[2,cc] = Cellsum
            cc +=1
             
            if Cellsum > ContentThreshold:
                BinarySelectionMap[j*SectionSize: (j+1)*SectionSize -1,i*SectionSize: (i+1)* SectionSize -1] = 255
                CellMap[j,i] = 1
        
        
    #%%  Superselection method to crop the image based on activated cells, to discard all space/black spots
    # Currently I set it up to keep ALL cells to avoid accidentally discarding useful data, but maybe it can use a fancier selection method
    # I have no idea if superselection is the proper name for this but I made it up in the flow, it sounds fancy. 
        
    # Again pre-allocating some space before counting the number of active cells in every row and column
    CellMapSumsH = np.zeros((CellsW,1))  
    CellMapSumsW = np.zeros((CellsH,1))    
    
    for i in range(0,CellsH):
        CellMapSumsW[i] = np.sum(CellMap[i,:])
        print(i)
        
    for i in range(0,CellsW):
        CellMapSumsH[i] = np.sum(CellMap[:,i])   
    
    
    #Create a heatmap of the cells to toy around with, not working better than cell superselection atm.
    CellHeatMap = CellMapSumsW + CellMapSumsH.T
    CellHeatMapimg = NormalizeMatrix(CellHeatMap,MaxGrayScale);
    
    #pre-allocate memory for the super selection
    SuperSelection = np.zeros(CellMap.shape)
    
    #pre-initialize some coordinates for the content selection box
    HighestX = 0
    HighestY = 0
    LowestX = img.shape[1]
    LowestY = img.shape[0]
    
    # Go through all cells and update the content selection boss on activated once to create a box which encompasses all active cells
    # Some leftover prints in here to help visualise the process, but not necessary. 
    for i in range(0,CellsH):
        for j in range(0,CellsW):
            if CellMapSumsH[j] > 1 and CellMapSumsW[i] > 1:
                LowH = np.max([0,i-R])
                MaxH = np.min([CellsH -1, i+R])
                LowW = np.max([0,j-R])
                MaxW = np.min([CellsW-1, j+R])
                
                # Cell ID is computed to grab the cell pixel data from CellInfo
                LowerCellID = LowW*CellsH + LowH
                LowerX = CellInfo[0,LowerCellID]
                LowerY = CellInfo[1,LowerCellID]
                HigherCellID = MaxW*CellsH + MaxH
                HigherX = CellInfo[0,HigherCellID]
                HigherY = CellInfo[1,HigherCellID]
                
                #print("Superselection from" , [LowerX, LowerY] , "to" , [HigherX,HigherY])
                
                if  HigherY > HighestY:
                    HighestY = HigherY
                   # print("New lowest point at " , [LowestX,LowestY])
                    
                if  HigherX > HighestX:
                    HighestX = HigherX
                   # print("New highest point at " , [HighestX,HighestY])
                    
                if  LowerY < LowestY:
                    LowestY = LowerY
                   # print("New lowest point at " , [LowestX,LowestY])
                    
                if LowerX < LowestX:
                    LowestX = LowerX
                   # print("New lowest point at " , [LowestX,LowestY])
    
    print(" Final superselection box from " , [LowestX,LowestY] , " to " , [HighestX, HighestY])
    
    
    #%% Some misc code to visualise the heatmap
    
    CellHeatMapVisual = np.zeros(img.shape)
    
    for i in range(0,CellsW):
        for j in range(0,CellsH):
            CellHeatMapVisual[j*SectionSize: (j+1)*SectionSize -1,i*SectionSize: (i+1)* SectionSize -1] = CellHeatMapimg[j,i];
    
    #crude filter to play around with filtering the heatmap, does not work as well... 
    #CellHeatMapVisual[CellHeatMapVisual < 1*np.mean(CellHeatMapVisual)] = 0
            
    #Cast it to uint8 so the image works.
    CellHeatMapVisual = CellHeatMapVisual.astype(np.uint8)
    
    #%% Displaying all the visuals
    
    # need to prepare some of the matrixes to be suitable for the imshow function
    BinarySelectionimg = BinarySelectionMap.astype(np.uint8)
    img = img.astype(np.uint8)
    
    SuperSelection = img[np.int(LowestY):np.int(HighestY), np.int(LowestX):np.int(HighestX)]
    
    Max = np.max(img)
    Min = np.min(img)
    
    print("Corrected image max and min: ",  Max , Min )
    
    
    # These things keep opening fullscreen and I'm not sure how to downsize them.
    
    cv2.imshow('Original', cv2.resize(imgo, (WindowW,WindowH)))
    cv2.imshow('Normalized', cv2.resize(img, (WindowW,WindowH)))
    cv2.imshow('Binary cell map', cv2.resize(BinarySelectionimg, (WindowW, WindowH)))
    cv2.imshow('Cell Heatmap', cv2.resize(CellHeatMapVisual, (WindowW, WindowH)))
    cv2.imshow('Extraction', SuperSelection)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return





#%% SETUP VARIABLES
SectionSize = 40 #px
Threshold = 30; # grayscale value threshold for binary selection, values below are ignored. (space is below!)
ContentThreshold = 0.3*SectionSize*SectionSize #min number of binary pixels within a cell required to flag it as active
R = 4; # Cell superselection range, larger makes the crop larger. Smaller makes it smaller, but risks losing some data
MaxGrayScale = 255 # Max uint8 value for the grayscale display.

# Image display window size bc fullscreen is annoying as hell
WindowW = 960
WindowH = 600
#images to analyse in a single run. replacing it each run was tiresome.
images = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
LazyOverride = 3; # override the loop without having to change the array of names

Nloops = np.min([len(images), LazyOverride])

for i in range(0,Nloops):
    PlayWithImage(images[i])


#%% 


