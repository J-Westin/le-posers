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
import matplotlib.pyplot as plt
from scipy import fftpack

def NormalizeMatrix(Matrix, rescale):
    "Normalize a matrix between 0 and rescale value"
    Max = np.max(Matrix)
    Matrix = np.round((Matrix/Max)*rescale)
    
    return Matrix

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap is necessary (linear scale shows only the central peak much higher than the amplitude at higher frequencies)
    plt.imshow(np.abs(im_fft), norm=LogNorm()) #sets the scale automatically
    plt.colorbar()

def PlayWithImage( imgName ):
    start = time.time()
    #Create a copy of the image after reading it to compare later.
    imgo = cv2.imread(imgName,0)
    img = imgo

    pxW = img.shape[0]
    pxH = img.shape[1]
    # Firstly normalize the image to make it easier to handle very dark photos in particular.
    img = NormalizeMatrix(img, MaxGrayScale)
    
    # Figure out how many cells the current resolution can support, currently not compensating for irregular sectionsizes
    CellsW = np.int(pxW/SectionSize)
    CellsH = np.int(pxH/SectionSize)
    
    # Pre-allocate some memory space
    CellInfoCluster = np.zeros((5,CellsH,CellsW)) #x,y,Content,ClusterID,state by coordinates
    CellNeighbourInfo = np.zeros((4,CellsH,CellsW)) # upper nbr info, right nbr info, bottom Nrb info, left nbr info
    
    #################### super hard threshold cut
    
    ###########################################################################
    # Insert of Tiberiu's corona analysis code
    #########################################
    
    
    ############################################################
    # Compute the 2d FFT of the input image
    ############################################################
    
    M, N = img.shape
#    print (M)
#    print (N)
    im_fft = fftpack.fft2(img) #Computes the 2D discrete aperiod fourier transform. Does NOT divide for M*N
    im_fft = im_fft / (M*N) #Dimensionality suggests to divide for total number of pixels (see literature)
    im_fft = np.fft.fftshift(im_fft)#To reconstruct the image, only one quadrant would be necessary. However, to view the patterns, is better to center in the origin the spectrum.    
    
    end = time.time()
    print("Initial Fourier runtime was " , end-start, "seconds")
#    plt.figure()
#    plot_spectrum(im_fft)#plots the spectrum in logaritmic scale normalized on the total number of pixels
#    plt.title('Normalized Fourier transform (log)')
    
    ############################################################
    # Filter in FFT
    ############################################################
    
    # In the lines following, we'll make a copy of the original spectrum and
    # truncate coefficients.
    
    # Define the fraction of coefficients (in each direction) we keep
    #keep_fraction = 0.5
    
    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()
    
    
    radiusmax=Rmax*min(M,N)/2 # radiuses of corona of filtering is a factor of the smaller dimension of the image, by choice
    radiusmin=Rmin*min(M,N)/2 
    
    mask = np.zeros(im_fft2.shape)
    maskx = np.square(np.linspace(-N/2,N/2,N))
    masky = np.square(np.linspace(-M/2,M/2,M))
    maskx.shape = (1,N)
    masky.shape = (M,1)
    mask = maskx
    mask = np.sqrt(maskx + masky)
    
    im_fft2[mask < radiusmin] = 0
    im_fft2[mask > radiusmax] = 0
    
#    plt.figure()
#    plot_spectrum(im_fft2)
#    plt.title('Filtered Spectrum')
    
    
    ############################################################
    # Reconstruct the final image
    ############################################################
    
    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(np.fft.ifftshift(im_fft2)).real 
    maxres=np.max(im_new)
#    plt.figure()
#    plt.imshow(im_new, plt.cm.gray)
#    plt.title('Reconstructed Image from Filtered Spectrum')
    
    
    im_new[im_new < 0.2*maxres] = 0
    im_new = NormalizeMatrix(im_new,255)
    im_new[im_new > Threshold/255] = 255
    im_new = im_new.astype(np.uint8)
    
    
    end = time.time()
    print("Fourier analysis runtime was " , end-start, "seconds")
    
    #cv2.imshow('Fourier Deconstruction', cv2.resize(im_new, (WindowW,WindowH)))

    
    # cc is a simple counter but it is also used as a Cell ID, to couple it to its origin x,y coordinates in CellInfo
    cc = 0
    for i in range(0,CellsW):
        for j in range(0,CellsH):
            Cellsum = 0
            Cell = im_new[j*SectionSize: (j+1)*SectionSize -1, i*SectionSize: (i+1)* SectionSize -1]
            BinarySelection = np.zeros(Cell.shape)
            BinarySelection[Cell>Threshold] = 1
            Cellsum = np.sum(BinarySelection)
            
            CellInfoCluster[0,j,i] = i*SectionSize
            CellInfoCluster[1,j,i] = j*SectionSize
            CellInfoCluster[2,j,i] = Cellsum
            CellInfoCluster[3,j,i] = cc
                
            cc +=1
    
    MaxCellSum = np.max(CellInfoCluster[2,:,:])
    CellInfoCluster[4,:,:][CellInfoCluster[2,:,:] > ContentThreshold*MaxCellSum] = 1
    CellInfoCluster[3,:,:][CellInfoCluster[4,:,:] == 0] = 0
        
   #%% Cluster identification of cells , maybe I could combine both loops into 1 with inverse coords...
    
    cc = 0 
    ClearMatrix = np.zeros((RClustering, RClustering));
    VisMatrix = np.zeros(img.shape)
    
    for x in range(0,CellsW):
        for y in range(0,CellsH):
            SubMatrix = ClearMatrix
            
            RangeUp = np.max([0,y-RClustering])
            RangeDown = np.min([CellsH-1,y+RClustering]) #Maybe add minus 1?
            RangeLeft = np.min([0,x-RClustering])
            RangeRight = np.max([CellsW-1,x+RClustering])
            
            SubMatrix = CellInfoCluster[3,(y-RangeUp):(y+RangeDown), (x-RangeLeft):(x+RangeRight)];
            
            if np.sum(SubMatrix[np.nonzero(SubMatrix)]) > 0:
                MinVal = np.min(SubMatrix[np.nonzero(SubMatrix)]);
                SubMatrix[SubMatrix > 0] = MinVal
                
                gu = np.max([0,y-1])
                gd = np.min([CellsH-1,y+1])
                gr = np.min([CellsW-1,x+1])
                gl = np.max([0,x-1])
                
                CellNeighbourInfo[0,y,x] = np.sum(CellInfoCluster[2, gu,(gl):(gr)])  /((1+(gr-gl))*ContentThreshold*MaxCellSum) # Up
                CellNeighbourInfo[1,y,x] = np.sum(CellInfoCluster[2, (gu):(gd), gr]) /((1+(gd-gu))*ContentThreshold*MaxCellSum)# Right
                CellNeighbourInfo[2,y,x] = np.sum(CellInfoCluster[2, gd,(gl):(gr)])  /((1+(gr-gl))*ContentThreshold*MaxCellSum)# Down
                CellNeighbourInfo[3,y,x] = np.sum(CellInfoCluster[2, (gu):gd,gl])  /((1+(gd-gu))* ContentThreshold*MaxCellSum)# Left
                
                CellInfoCluster[3,(y-RangeUp):(y+RangeDown), (x-RangeLeft):(x+RangeRight)] = SubMatrix;
                VisMatrix[y*SectionSize: (y+1)*SectionSize -1,x*SectionSize: (x+1)* SectionSize -1] = CellInfoCluster[3,y,x]
   
               
    #%%  Identify the NR of individual clusters
    X , Y = np.nonzero(CellInfoCluster[3,:,:])
    ClusterIDs = set(CellInfoCluster[3,X,Y])
    ClusterIDs = list(ClusterIDs)
    Clustercount = len(ClusterIDs)
    print("Different clusters counted:" , Clustercount,   " with ids", ClusterIDs) 
    
    
    #%% Debugging tool/visualise the cluster cells
    VisMatrix = NormalizeMatrix(VisMatrix,255);
    cv2.imshow('Cell cluster ID', cv2.resize(VisMatrix.astype(np.uint8), (WindowW,WindowH)))
    
    
    #%% Case handling and preparing to export the data.
    
    
    # Multidimensional numeric array to be able to export any amount of images in a single variable
    ExportImages = np.zeros((Clustercount,TargetImgH,TargetImgW))
    HeaderCoordinates = np.zeros((2,Clustercount)) # contains H and W coordinates of img. start
    
    if (Clustercount > 1):
        print("!!!! Potential problematic image; " , imgName, "!!!! Multiple image clusters identified!, Returning full image" )
        ExportImages[1,:,:] = cv2.resize(img, (TargetImgW, TargetImgH))
    else:
        #Little bit of legacy code in the for loop, can be removed later.
        for N in range(0,Clustercount):
            # First find the coordinates in pixels for the crop
            cid = ClusterIDs[N]
            lc = CellInfoCluster[3,:,:] 
            H , W = np.where(lc == cid)
            #Min is the upper left coordinate
            MinCoordH = np.min(H) -1
            MinCoordW = np.min(W) -1
            MinCoordH = np.max([0,MinCoordH]);
            MinCoordW = np.max([0,MinCoordW]);
            # Max is the upper right coordinate
            MaxCoordH = np.max([0, np.max(H)])
            MaxCoordW = np.min([CellsW-1, np.max(W)])

            width = MaxCoordW - MinCoordW
            height = MaxCoordH - MinCoordH
            
            ## Compute the dynamic cropping box from content values in neighbouring cells            
            RU = np.floor(-np.cos(np.sum(CellNeighbourInfo[0,MinCoordH,MinCoordW:MaxCoordW])/width)*R/4 + R).astype(int)
            RL = np.floor(-np.cos(np.sum(CellNeighbourInfo[3,MinCoordH:MaxCoordH,MinCoordW])/height)*R/4 + R).astype(int)
            RR = np.floor(-np.cos(np.sum(CellNeighbourInfo[1,MinCoordH:MaxCoordH,MaxCoordW])/height)*R/4 + R).astype(int)
            RD = np.floor(-np.cos(np.sum(CellNeighbourInfo[2,MaxCoordH,MinCoordW:MaxCoordW])/width)*R/4 + R).astype(int)
            
            MinCoordH = np.max([0,MinCoordH*SectionSize - RU])
            MinCoordW = np.max([0,MinCoordW*SectionSize - RL ])
            MaxCoordH = np.min([img.shape[0],MaxCoordH*SectionSize + SectionSize + RD])
            MaxCoordW = np.min([img.shape[1],MaxCoordW*SectionSize + SectionSize + RR])
            
            print("selection for cluster ", cid , "from coordinates " , MinCoordW,",",  MinCoordH , "to ", MaxCoordW , ", " , MaxCoordH )
            
            ClusterImg = img[MinCoordH:MaxCoordH, MinCoordW: MaxCoordW]
            
            #print("Cluster", cid , "has size " , ClusterImg.shape[1],",",  ClusterImg.shape[0])
            
            # Image resizing to the specified Height and Width
            CIH = MaxCoordH - MinCoordH
            CIW = MaxCoordW - MinCoordW
            
            #HC contains H and W coordinates from a top-left origin frame
            HeaderCoordinates[0,N] = MaxCoordH
            HeaderCoordinates[1,N] = MinCoordW    
            
            # Resize by filling the smallest dimension with empty pixels, to avoid stretching the img. 
            # Always put empty pixels above or to the right, so that origin of the new img is MinCoordW,MinCoordH
            
            #cv2.imshow('raw cutout img', ClusterImg)
            
            if CIH < CIW : # Need to fill in H direction before resizing
                # Target image ratio based on the aspect ratio desired
                TIW = CIW
                TIH = TIW/AspectRatio
                TIH = TIH.astype(int)
                TIW = TIW.astype(int)
                
                diff = CIW-CIH
                newimg = np.zeros((TIH,TIW))
                if MinCoordH == 0:
                    newimg[0:(TIH-diff), 0:TIW] = ClusterImg
                else:
                    newimg[diff:TIH, 0:TIW] = ClusterImg
                ClusterImg = newimg
                ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
            elif CIW < CIH : # need to fill in the W direction
                # Target image ratio based on the aspect ratio desired
                TIH = CIH
                TIW = AspectRatio*TIH
                TIH = TIH.astype(int)
                TIW = TIW.astype(int)
                
                diff = CIH-CIW
                newimg = np.zeros((TIH,TIH))
                if MaxCoordW == CellsW -1:
                    newimg[0:TIH, (TIW-CIW):TIW] = ClusterImg
                else:
                    newimg[0:TIH, 0:CIW] = ClusterImg
                ClusterImg = newimg
                ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
            else:
                ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
            
            ExportImages[N,:,:] = ClusterImg
            
            for N in range(0,Clustercount):
                namestring = "Cluster Image " + str(N)
                cv2.imshow(namestring , ExportImages[N,:,:].astype(np.uint8))
        
        
    end = time.time()
    print("total runtime was " , end-start, "seconds")
    #%% Displaying all the visuals
    
    # need to prepare some of the matrixes to be suitable for the imshow function
    #BinarySelectionimg = BinarySelectionMap.astype(np.uint8)
    img = img.astype(np.uint8)

    #cv2.imshow('Original', cv2.resize(imgo, (WindowW,WindowH)))
    cv2.imshow('Normalized', cv2.resize(img, (WindowW,WindowH)))
    #cv2.imshow('Binary cell map', cv2.resize(BinarySelectionimg, (WindowW, WindowH)))    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return





#%% SETUP VARIABLES
    
#%% NEED TO MERGE CELLS WITH SMALLER DISTANCE THAN 1!

# Can also be used as input variables to run as indep. function    
 
SectionSize = 120 #px
Threshold = 20; # grayscale value threshold for binary selection, values below are ignored. (space is below!)
ContentThreshold = 0.6 # % of pixels required to be in a selection cell 
R = 400; # Cell superselection range in px, larger makes the crop larger
MaxGrayScale = 255 # Max uint8 value for the grayscale display.
RClustering = 2

## Fourier setup
Rmax = 0.54
Rmin = 0.35

TargetImgW = 500
TargetImgH = 500

# Image display window size bc fullscreen is annoying as hell
WindowW = 960
WindowH = 600
#images to analyse in a single run. replacing it each run was tiresome.
images = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
LazyOverride = 1; # override the loop without having to change the array of names

Nloops = np.min([len(images), LazyOverride])

AspectRatio = TargetImgW / TargetImgH
 

for i in range(0,Nloops):
    PlayWithImage(images[i])


#%% GENERAL NOTES
# Solar panels are problematic! need to find the SP frequency band as well...

