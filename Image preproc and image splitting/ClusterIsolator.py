# -*- coding: utf-8 -*-
"""
This code should work with most if not all configurations, but it might have problems if a object separates into multiple clusters
Need to find a way to detect and avoid that, but being lenient with the Threshold also seems to work so far

21/02/19 avg runtime is 0.35s per image now, could prob be optimized much more.
"""

#import tensorflow as tf
import numpy as np
import cv2 
import time

def NormalizeMatrix(Matrix, rescale):
    "Normalize a matrix between 0 and rescale value"
    Max = np.max(Matrix)
    Matrix = np.round((Matrix/Max)*rescale)
    
    return Matrix

def PlayWithImage( img ):
    start = time.time()
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
    CellInfo = np.zeros((4,np.int(TCells))) #x,y,Content,ClusterID
    CellInfoCluster = np.zeros((5,CellsH,CellsW)); #x,y,Content,ClusterID,state by coordinates
    CellMap = np.zeros((CellsH,CellsW))    # Binary map of the cells
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
            
            CellInfoCluster[0,j,i] = i*SectionSize
            CellInfoCluster[1,j,i] = j*SectionSize
            CellInfoCluster[2,j,i] = Cellsum
            CellInfo[0,cc] = i*SectionSize
            CellInfo[1,cc] = j*SectionSize
            CellInfo[2,cc] = Cellsum
            

            
             
            if Cellsum > ContentThreshold:
                BinarySelectionMap[j*SectionSize: (j+1)*SectionSize -1,i*SectionSize: (i+1)* SectionSize -1] = 255
                CellMap[j,i] = 1
                CellInfo[3,cc] = cc
                CellInfoCluster[3,j,i] = cc
                CellInfoCluster[4,j,i] = 1
                
            cc +=1
        
        
        
        
   #%% Cluster identification of cells , maybe I could combine both loops into 1 with inverse coords...
    cc = 0
    VisMatrix = np.zeros(img.shape)
    for i in range(0,CellsW):
       for j in range(0,CellsH):
           
           Neighbours = np.zeros(8)
           
           if j >= 1:
               if i >= 1:
                   Neighbours[0] = CellInfoCluster[3,j-1,i-1]
               Neighbours[1] = CellInfoCluster[3,j-1,i]
               if i < CellsW -1:
                   Neighbours[2] = CellInfoCluster[3,j-1,i+1]
           if i >= 1:
               Neighbours[3] = CellInfoCluster[3,j,i-1]
           if i < CellsW -1:
               Neighbours[4] = CellInfoCluster[3,j,i+1]
           if j < CellsH -1:
               if i>= 1:
                   Neighbours[5] = CellInfoCluster[3,j+1,i-1]
                   
               Neighbours[6] = CellInfoCluster[3,j+1,i]
               
               if i < CellsW -1:
                   Neighbours[7] = CellInfoCluster[3,j+1, i+1]
           
           Neighbourset = set(Neighbours[np.nonzero(Neighbours)]) #identify unique cluster IDs disregard 0
           Neighbourcount = len(Neighbourset) # Count the number of unique cluster IDs 
           Neighbourset = list(Neighbourset)
           Neighbours = Neighbours.tolist();
           

               
           if Neighbourcount > 0 and CellInfoCluster[4,j,i] == 1: 
               
               Uniquecnt = np.zeros(Neighbourcount) #contains the NR of identical cluster IDs per cluster
               #print("Nbrcnt ", Neighbourcount)
               #print("Nbrset ", Neighbourset)
               for c in range(0,Neighbourcount):
                   Uniquecnt[c] = Neighbours.count(Neighbourset[c])
                   
               #legacy code, just picking the smalles cid works better
#               if len(set(Uniquecnt[np.nonzero(Uniquecnt)])) > 1:
#                   Uniquelst = Uniquecnt.tolist()
#                   CellInfoCluster[3,j,i] = Neighbourset[Uniquelst.index(np.max(Uniquecnt))]
#               else:
#                   CellInfoCluster[3,j,i] = np.min(Neighbourset);
                   
               CellInfoCluster[3,j,i] = np.min(Neighbourset);
#             
#           
           VisMatrix[j*SectionSize: (j+1)*SectionSize -1,i*SectionSize: (i+1)* SectionSize -1] = CellInfoCluster[3,j,i]
   
    # A reverse run has to be done to work out a few kinks that can occur with certain geometries
    for i in range(CellsW-1,-1,-1):
        for j in range(CellsH-1,-1,-1):
            
               Neighbours = np.zeros(8)
               
               if j >= 1:
                   if i >= 1:
                       Neighbours[0] = CellInfoCluster[3,j-1,i-1]
                   Neighbours[1] = CellInfoCluster[3,j-1,i]
                   if i < CellsW -1:
                       Neighbours[2] = CellInfoCluster[3,j-1,i+1]
               if i >= 1:
                   Neighbours[3] = CellInfoCluster[3,j,i-1]
               if i < CellsW -1:
                   Neighbours[4] = CellInfoCluster[3,j,i+1]
               if j < CellsH -1:
                   if i>= 1:
                       Neighbours[5] = CellInfoCluster[3,j+1,i-1]
                   Neighbours[6] = CellInfoCluster[3,j+1,i]
                   if i < CellsW -1:
                       Neighbours[7] = CellInfoCluster[3,j+1, i+1]
               
               Neighbourset = set(Neighbours[np.nonzero(Neighbours)]) #identify unique cluster IDs disregard 0
               Neighbourcount = len(Neighbourset) # Count the number of unique cluster IDs 
               Neighbourset = list(Neighbourset)
               Neighbours = Neighbours.tolist();
               
    
                   
               if Neighbourcount > 0 and CellInfoCluster[4,j,i] == 1: 
                   
                   Uniquecnt = np.zeros(Neighbourcount) #contains the NR of identical cluster IDs per cluster
                   #print("Nbrcnt ", Neighbourcount)
                   #print("Nbrset ", Neighbourset)
                   for c in range(0,Neighbourcount):
                       Uniquecnt[c] = Neighbours.count(Neighbourset[c])
                       
#                   if len(set(Uniquecnt[np.nonzero(Uniquecnt)])) > 1:
#                       Uniquelst = Uniquecnt.tolist()
#                       CellInfoCluster[3,j,i] = Neighbourset[Uniquelst.index(np.max(Uniquecnt))]
#                   else:
#                       CellInfoCluster[3,j,i] = np.min(Neighbourset);
                   
                   CellInfoCluster[3,j,i] = np.min(Neighbourset);
    #             
    #           
               VisMatrix[j*SectionSize: (j+1)*SectionSize -1,i*SectionSize: (i+1)* SectionSize -1] = CellInfoCluster[3,j,i]
               
               
    #%% 
    X , Y = np.nonzero(CellInfoCluster[3,:,:])
    ClusterIDs = set(CellInfoCluster[3,X,Y])
    ClusterIDs = list(ClusterIDs)
    Clustercount = len(ClusterIDs)
    print("Different clusters counted:" , Clustercount,   " with ids", ClusterIDs) 
    
    #BinarySelectionMap[0:15, 0] = 255
    VisMatrix = NormalizeMatrix(VisMatrix,255);
    cv2.imshow('Cell cluster ID', cv2.resize(VisMatrix.astype(np.uint8), (WindowW,WindowH)))
    
    # Multidimensional numeric array to be able to export any amount of images in a single variable
    ExportImages = np.zeros((Clustercount,TargetImgH,TargetImgW))
    HeaderCoordinates = np.zeros((2,Clustercount)) # contains H and W coordinates of img. start
    
    for N in range(0,Clustercount):
        # First find the coordinates in pixels for the crop
        cid = ClusterIDs[N]
        lc = CellInfoCluster[3,:,:] 
        H , W = np.where(lc == cid)
        #Min is the upper left coordinate
        MinCoordH = np.max(H) + R 
        MinCoordW = np.min(W) - R 
        # Max is the upper right coordinate
        MaxCoordH = np.max([0, np.min(H) - R])
        MaxCoordW = np.min([CellsW-1, np.max(W) + R])
        
        MinCoordH = MinCoordH*SectionSize - SectionSize
        MinCoordW = MinCoordW*SectionSize
        MaxCoordH = MaxCoordH*SectionSize 
        MaxCoordW = MaxCoordW*SectionSize + SectionSize
        
        print("selection for cluster ", cid , "from coordinates " , MinCoordW,",",  MinCoordH , "to ", MaxCoordW , ", " , MaxCoordH )
        
        ClusterImg = img[MaxCoordH:MinCoordH, MinCoordW: MaxCoordW]
        
        # Image resizing to the specified Height and Width
        CIH = MinCoordH - MaxCoordH
        CIW = MaxCoordW - MinCoordW
        
        #HC contains H and W coordinates from a top-left origin frame
        HeaderCoordinates[0,N] = MaxCoordH
        HeaderCoordinates[1,N] = MinCoordW

        
        # Resize by filling the smallest dimension with empty pixels, to avoid stretching the img. 
        # Always put empty pixels above or to the right, so that origin of the new img is MinCoordW,MinCoordH
        
        if CIH < CIW: # Need to fill in H direction before resizing
            # Target image ratio based on the aspect ratio desired
            TIW = CIW
            TIH = TIW/AspectRatio
            TIH = TIH.astype(int)
            TIW = TIW.astype(int)
            
            diff = CIW-CIH
            newimg = np.zeros((TIH,TIW))
            newimg[diff:TIH, 0:TIW] = ClusterImg
            ClusterImg = newimg
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        elif CIW < CIH: # need to fill in the W direction
            # Target image ratio based on the aspect ratio desired
            TIH = CIH
            TIW = AspectRatio*TIH
            TIH = TIH.astype(int)
            TIW = TIW.astype(int)
            
            diff = CIH-CIW
            newimg = np.zeros((TIH,TIH))
            newimg[0:TIH, 0:CIW] = ClusterImg
            ClusterImg = ClusterImg
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        else:
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        
        
        ExportImages[N,:,:] = ClusterImg
        

        
    end = time.time()
    print("total runtime was " , end-start, "seconds")
    #%% Displaying all the visuals
    
    # need to prepare some of the matrixes to be suitable for the imshow function
    BinarySelectionimg = BinarySelectionMap.astype(np.uint8)
    img = img.astype(np.uint8)

    Max = np.max(img)
    Min = np.min(img)
    
    print("Corrected image max and min: ",  Max , Min )
    
    for N in range(0,Clustercount):
        namestring = "Cluster Image " + str(N)
        cv2.imshow(namestring , ExportImages[N,:,:].astype(np.uint8))
    
    
    # These things keep opening fullscreen and I'm not sure how to downsize them.
    
    #cv2.imshow('Original', cv2.resize(imgo, (WindowW,WindowH)))
    #cv2.imshow('Normalized', cv2.resize(img, (WindowW,WindowH)))
    cv2.imshow('Binary cell map', cv2.resize(BinarySelectionimg, (WindowW, WindowH)))
    #cv2.imshow('Cell Heatmap', cv2.resize(CellHeatMapVisual, (WindowW, WindowH)))
    #cv2.imshow('Extraction', SuperSelection)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return





#%% SETUP VARIABLES

# Can also be used as input variables to run as indep. function    

SectionSize = 40 #px
Threshold = 20; # grayscale value threshold for binary selection, values below are ignored. (space is below!)
ContentThreshold = 0.3*SectionSize*SectionSize #min number of binary pixels within a cell required to flag it as active
R = 4; # Cell superselection range, larger makes the crop larger. Smaller makes it smaller, but risks losing some data
MaxGrayScale = 255 # Max uint8 value for the grayscale display.


TargetImgW = 500
TargetImgH = 500

# Image display window size bc fullscreen is annoying as hell
WindowW = 960
WindowH = 600
#images to analyse in a single run. replacing it each run was tiresome.
images = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
LazyOverride = 5; # override the loop without having to change the array of names

Nloops = np.min([len(images), LazyOverride])

AspectRatio = TargetImgW / TargetImgH
 

for i in range(0,Nloops):
    PlayWithImage(images[i])


#%% 


