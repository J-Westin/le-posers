# -*- coding: utf-8 -*-
"""
This code should work with most if not all configurations, but it might have problems if a object separates into multiple clusters
Need to find a way to detect and avoid that, but being lenient with the Threshold also seems to work so far

23/03/19: Classified as version 1.0. Average runtime is a second per image (which is a lot)
Overcropping is a problem when the satellite is close to the camera...
Earth cloud patterns with the same fourier frequency as the satellite components throw off the clustering, resulting in missed cropping opportunities. 

.1A subbranch, going to experiment with Canny edge detection
.2A Subbranch, Incorporated Canny edge detection, cleanup of 1A
.3A subbranch, Downsamples the image to significantly improve performance, but now needs some more tweaking....

FINE-TUNING STILL IN PROGRESS, COULD DO WITH A BIT MORE...
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
    Matrix = (Matrix/Max)*rescale
#    Matrix //= Max
#    Matrix *= rescale
    Matrix.astype(np.int)
    
    return Matrix


def HyperNormalizeMatrix(Matrix, rescale):
    "Normalize a matrix between 0 and rescale value"
#    Matrix = Matrix.astype(np.float64)
#    Max = np.max(Matrix).astype(np.float64)
#    Mean = np.mean(Matrix[Matrix > 0.2*Max]).astype(np.float64)
#    
#    Matrix[Matrix > 0.08*Max] += int(0.5*(Max-Mean)/Max * Mean)
#    Matrix[Matrix > 0.90*Max] -= int(0.5*(Max-Mean)/Max * Mean)
#    
#    Max = np.max(Matrix)
#    Matrix = (Matrix/Max)*rescale
#    Matrix.astype(np.uint8)
    
    "Normalize a matrix between 0 and rescale value"
    Matrix = Matrix.astype(np.float64)
    Max = np.max(Matrix)
    Mean = np.mean(Matrix[Matrix > 0.2*Max])
    
    Matrix[Matrix > 0.08*Max] += 0.5*((Max-Mean)/Max) * Mean
    Matrix[Matrix > 0.9*Max] -= 0.5*((Max-Mean)/Max) * Mean
    
    Max = np.max(Matrix)
    Matrix = (Matrix/Max)*rescale
    Matrix.astype(np.int)
    
    return Matrix


def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap is necessary (linear scale shows only the central peak much higher than the amplitude at higher frequencies)
    plt.imshow(np.abs(im_fft), norm=LogNorm()) #sets the scale automatically
    plt.colorbar()


def PictureSnipSnipper( imgName , TargetImgW, TargetImgH):
    #%% Some setup variables:
    # Can also be used as input variables to run as indep. function    
 
    SectionSize = 60 #px
    Threshold = 20; # grayscale value threshold for binary selection, values below are ignored. (space is below!)
    ContentThreshold = 0.45 # % of pixels required to be in a selection cell 
    R = 400; # Cell superselection range in px, larger makes the crop larger
    MaxGrayScale = int(250) # Max uint8 value for the grayscale display.
    RClustering = 5
    BorderPixClearance = 20;
    
    AspectRatio = TargetImgW / TargetImgH
    CannyLow = 200
    CannyHigh = 530
    Downscalefactor = 2
    CannyContent = 0.25

    
    ## Fourier setup # commented segments work well for full res
    Rmax = 0.78#0.2#0.54
    Rmin = 0.54#0.17#0.35

    radiusMuffleMax = 0.7
    
    # Image display window size bc fullscreen is annoying as hell
    WindowW = 960
    WindowH = 600
    
    
    
    
    #Create a copy of the image after reading it to compare later.
    imgo = cv2.imread(imgName,0)
    img = imgo  
         
    start = time.time()
    if imgo is None:
        print("Requested image does not exist!!! Very sad :( ")
        return
    
    pxH = img.shape[0]
    pxW = img.shape[1]
    
    # Firstly normalize the image to make it easier to handle very dark photos in particular.
    imgN = NormalizeMatrix(img,MaxGrayScale)
    img = HyperNormalizeMatrix(img, MaxGrayScale)
    
    # Figure out how many cells the current resolution can support, currently not compensating for irregular sectionsizes
    CellsW = int(pxW/SectionSize)
    CellsH = int(pxH/SectionSize)
    
    # Pre-allocate some memory space
    CellInfoCluster = np.zeros((6,CellsH,CellsW)) #x,y,Content,ClusterID,state by coordinates, canny edge sum
    CellNeighbourInfo = np.zeros((4,CellsH,CellsW)) # upper nbr info, right nbr info, bottom Nrb info, left nbr info
    
    w = int(pxW/Downscalefactor)
    h = int(pxH/Downscalefactor)
    imgf = cv2.resize(img,(w,h))
    end = time.time()
    print("Secondary runtime was " , end-start, "seconds")
    #################### super hard threshold cut
    
    ###########################################################################
    # Insert of Tiberiu's corona analysis code
    #########################################
    
    
    ############################################################
    # Compute the 2d FFT of the input image
    ############################################################
    
    M, N = imgf.shape
#    print (M)
#    print (N)
    im_fft = fftpack.fft2(imgf) #fftpack.fft2(img) #Computes the 2D discrete aperiod fourier transform. Does NOT divide for M*N
    im_fft = im_fft / (M*N) #Dimensionality suggests to divide for total number of pixels (see literature)
    im_fft = fftpack.fftshift(im_fft)#  np.fft.fftshift(im_fft)#To reconstruct the image, only one quadrant would be necessary. However, to view the patterns, is better to center in the origin the spectrum.    
    
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
    
    # Muffle image contains a smaller part of the spectrum to reduce noise, increases canny edge detection perf.
    im_fftmuffle = im_fft.copy()
    
    Minimum = min(M,N)/2
    radiusmax=Rmax*Minimum # radiuses of corona of filtering is a factor of the smaller dimension of the image, by choice
    radiusmin=Rmin*Minimum 
    radiusMuffleMax = radiusMuffleMax*Minimum
    
    mask = np.zeros(im_fft.shape)
    maskx = np.square(np.linspace(-N/2,N/2,N))
    masky = np.square(np.linspace(-M/2,M/2,M))
    maskx.shape = (1,N)
    masky.shape = (M,1)
    mask = maskx
    mask = np.sqrt(maskx + masky)
    
    im_fft[mask < radiusmin] = 0
    im_fft[mask > radiusmax] = 0
    im_fftmuffle [mask > radiusMuffleMax] = 0
#    plt.figure()
#    plot_spectrum(im_fft2)
#    plt.title('Filtered Spectrum')
    
    
    ############################################################
    # Reconstruct the final image
    ############################################################
    
    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(fftpack.ifftshift(im_fft)).real 
    maxres=np.max(im_new)

    im_new[im_new < 0.2*maxres] = 0
    im_new = NormalizeMatrix(im_new,MaxGrayScale)
    im_new[im_new > Threshold/MaxGrayScale] = MaxGrayScale
    im_new = im_new.astype(np.uint8)
    
    im_muffle = fftpack.ifft2(fftpack.ifftshift(im_fftmuffle)).real 
    maxmuff = np.max(im_muffle)

    im_muffle[im_muffle < 0.05*maxmuff] = 0
    im_muffle = NormalizeMatrix(im_muffle,MaxGrayScale)
    im_muffle = im_muffle.astype(np.uint8)
    
    
    end = time.time()
    print("Fourier analysis runtime was " , end-start, "seconds")
    CannyMuff = cv2.Canny(im_muffle,CannyLow,CannyHigh)
#    CannyMuff = cv2.resize(CannyMuff,(pxW,pxH))
#    im_new = cv2.resize(im_new,(pxW,pxH))
#    im_muffle = cv2.resize(im_muffle,(pxW,pxH))
    im_new = im_new.astype(np.uint8)
    im_muffle = im_muffle.astype(np.uint8)
    print(np.max(im_muffle))
    #CannyEdge = cv2.Canny(img.astype(np.uint8),150,300)
    
    
    
    
    
    im_CannyFourier = NormalizeMatrix(im_new + CannyMuff,255)
    im_CannyFourier[0:BorderPixClearance, :] = 0
    im_CannyFourier[(pxH-BorderPixClearance) : (pxH-1),:] = 0
    im_CannyFourier[:,0:BorderPixClearance] = 0
    im_CannyFourier[:,(pxW-BorderPixClearance):(pxW-1)] = 0
    im_new = im_CannyFourier


    
#    cv2.imshow('Canny Edge detection', cv2.resize(CannyEdge, (WindowW, WindowH)))          
#    cv2.imshow('Fourier Deconstruction', cv2.resize(im_new, (WindowW,WindowH)))
    cv2.imshow('Combined fourier + muffled Canny', cv2.resize(im_CannyFourier, (WindowW,WindowH)))
#    cv2.imshow('Cloud detection', cv2.resize(im_cloud, (WindowW,WindowH)))
    cv2.imshow('Muffled Image', cv2.resize(im_muffle, (WindowW,WindowH)))
#    cv2.imshow('Muffled Image Canny', cv2.resize(CannyMuff, (WindowW,WindowH)))


    # cc is a simple counter but it is also used as a Cell ID, to couple it to its origin x,y coordinates in CellInfo
    cc = 0
   # ECell = np.zeros((SectionSize-1, SectionSize-1))
    SectionSize = int(SectionSize/Downscalefactor)
    for i in range(0,CellsW):
        for j in range(0,CellsH):
            Cellsum = 0
            Cell = im_new[j*SectionSize:((j+1)*SectionSize -1), i*SectionSize:((i+1)* SectionSize -1)]
            BinarySelection = np.zeros((SectionSize-1, SectionSize-1))
            BinarySelection[Cell>Threshold] = 1
            Cellsum = np.sum(BinarySelection)
            
            CannySum = 0
            Cell = CannyMuff[j*SectionSize:((j+1)*SectionSize -1), i*SectionSize: ((i+1)* SectionSize -1)]
            BinarySelection2 = np.zeros((SectionSize-1, SectionSize-1))
            BinarySelection2[Cell>Threshold] = 1
            CannySum = np.sum(BinarySelection2)
            
#            array = [i*SectionSize,j*SectionSize,Cellsum,cc,0,CannySum]
#            CellInfoCluster[:,j,i] = array
            CellInfoCluster[0,j,i] = i*SectionSize
            CellInfoCluster[1,j,i] = j*SectionSize
            CellInfoCluster[2,j,i] = Cellsum
            CellInfoCluster[3,j,i] = cc
            CellInfoCluster[5,j,i] = CannySum
            
            
            cc +=1
    
    MaxCellSum = np.max(CellInfoCluster[2,:,:])
    CellInfoCluster[4,:,:][CellInfoCluster[2,:,:] > ContentThreshold*MaxCellSum] = 1
    CellInfoCluster[4,:,:][CellInfoCluster[5,:,:] < CannyContent*ContentThreshold*MaxCellSum] = 0
    CellInfoCluster[3,:,:][CellInfoCluster[4,:,:] == 0] = 0
    
        #%% Debugging tool/visualise the cluster cells
#    VisMatrix = np.zeros(imgo.shape)
#    for x in range(0,CellsW-1):
#        for y in range(0,CellsH-1):
#            if CellInfoCluster[4,y,x] == 1:
#                VisMatrix[y*SectionSize: (y+1)*SectionSize -1,x*SectionSize: (x+1)* SectionSize -1] = CellInfoCluster[3,y,x]
#    
#    VisMatrix = HyperNormalizeMatrix(VisMatrix,255);
#    VisMatrix = VisMatrix.astype(np.uint8);
#    cv2.imshow('Cell cluster ID before sublimation', cv2.resize(VisMatrix, (WindowW,WindowH)))
        
   #%% Cluster identification of cells , maybe I could combine both loops into 1 with inverse coords...
    
    cc = 0 
    #ClearMatrix = np.zeros((2*RClustering, 2*RClustering));
    
    
    stepsize = 1# np.floor(RClustering/2).astype(np.uint)
    cc = 0
    ss = cc
    for x in range(RClustering,CellsW-RClustering , stepsize):
        for y in range(RClustering,CellsH-RClustering , stepsize):
            
            SubMatrix = CellInfoCluster[3,(y-RClustering):(y+RClustering), (x-RClustering):(x+RClustering)];
            cc += 1
            if np.sum(SubMatrix[np.nonzero(SubMatrix)]) > 0:
                
                MinVal = np.min(SubMatrix[np.nonzero(SubMatrix)]);
                SubMatrix[np.nonzero(SubMatrix)] = MinVal
                
                unique, counts = np.unique(SubMatrix[np.nonzero(SubMatrix)], return_counts=True)
                
                MinVal = unique[counts == max(counts)]
                
                gu = np.max([0,y-1])
                gd = np.min([CellsH-1,y+1])
                gr = np.min([CellsW-1,x+1])
                gl = np.max([0,x-1])
                
                CellNeighbourInfo[0,y,x] = np.sum(CellInfoCluster[2, gu,(gl):(gr)])  /((1+(gr-gl))*ContentThreshold*MaxCellSum) # Up
                CellNeighbourInfo[1,y,x] = np.sum(CellInfoCluster[2, (gu):(gd), gr]) /((1+(gd-gu))*ContentThreshold*MaxCellSum)# Right
                CellNeighbourInfo[2,y,x] = np.sum(CellInfoCluster[2, gd,(gl):(gr)])  /((1+(gr-gl))*ContentThreshold*MaxCellSum)# Down
                CellNeighbourInfo[3,y,x] = np.sum(CellInfoCluster[2, (gu):gd,gl])  /((1+(gd-gu))* ContentThreshold*MaxCellSum)# Left
                
                #CellInfoCluster[3,(y-RangeUp):(y+RangeDown), (x-RangeLeft):(x+RangeRight)] = SubMatrix;
                CellInfoCluster[3,(y-RClustering):(y+RClustering), (x-RClustering):(x+RClustering)] = SubMatrix;
                #VisMatrix[y*SectionSize: (y+1)*SectionSize -1,x*SectionSize: (x+1)* SectionSize -1] = CellInfoCluster[3,y,x]
                ss += 1
   
    print ("total submatrix evaluations: ", cc, " of which " , ss, " triggered the selection step"  )
    #%%  Identify the NR of individual clusters
    X , Y = np.nonzero(CellInfoCluster[3,:,:])
    ClusterIDs = set(CellInfoCluster[3,X,Y])
    ClusterIDs = list(ClusterIDs)
    Clustercount = len(ClusterIDs)
    print("Different clusters counted:" , Clustercount,   " with ids", ClusterIDs) 
    
    SectionSize = int(SectionSize*Downscalefactor)
    #%% Debugging tool/visualise the cluster cells
    VisMatrix = np.zeros(imgo.shape)
    for x in range(0,CellsW-1):
        for y in range(0,CellsH-1):
            if CellInfoCluster[4,y,x] == 1:
                VisMatrix[y*SectionSize: (y+1)*SectionSize -1,x*SectionSize: (x+1)* SectionSize -1] = CellInfoCluster[3,y,x]
    
    VisMatrix = HyperNormalizeMatrix(VisMatrix,255);
    VisMatrix = VisMatrix.astype(np.uint8);
    cv2.imshow('Cell cluster ID', cv2.resize(VisMatrix, (WindowW,WindowH)))
    
    
    #%% Case handling and preparing to export the data.
    
    
    
    MinCoordH = 0
    MinCoordW = 0
    MaxCoordH = 0
    MaxCoordW = 0
    # Multidimensional numeric array to be able to export any amount of images in a single variable
    ExportImage = np.zeros((1,TargetImgH,TargetImgW))
    HeaderCoordinates = np.zeros((2,1)) # contains H and W coordinates of img. start
    
    if (Clustercount > 1):
        for i in range(0,Clustercount):
            CID = ClusterIDs[i]
            a = CellInfoCluster[3,:,:] == CID
            logics = np.sum(CellInfoCluster[4,a])
            if logics == 1 :
                CellInfoCluster[4,a] = 0
                CellInfoCluster[3,a] = 0
                print("Removed lone wolf cell!")
            
        
    X , Y = np.nonzero(CellInfoCluster[3,:,:])
    ClusterIDs = set(CellInfoCluster[3,X,Y])
    ClusterIDs = list(ClusterIDs)
    Clustercount = len(ClusterIDs)
    print("Different clusters counted:" , Clustercount,   " with ids", ClusterIDs) 
    
    
    
    if (Clustercount != 1):
        print("!!!! Potential problematic image; " , imgName, "!!!! Multiple image clusters identified!, Returning full image" )
        ExportImage[0,:,:] = cv2.resize(img, (TargetImgW, TargetImgH))
    else:
        # First find the coordinates in pixels for the crop
        cid = ClusterIDs[0]
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
        RU = np.floor(-np.cos(np.sum(CellNeighbourInfo[0,MinCoordH,MinCoordW:MaxCoordW])/width)*R/2 + R).astype(int)
        RL = np.floor(-np.cos(np.sum(CellNeighbourInfo[3,MinCoordH:MaxCoordH,MinCoordW])/height)*R/2 + R).astype(int)
        RR = np.floor(-np.cos(np.sum(CellNeighbourInfo[1,MinCoordH:MaxCoordH,MaxCoordW])/height)*R/2 + R).astype(int)
        RD = np.floor(-np.cos(np.sum(CellNeighbourInfo[2,MaxCoordH,MinCoordW:MaxCoordW])/width)*R/2 + R).astype(int)
        
        MinCoordH = np.max([0,MinCoordH*SectionSize - RU])
        MinCoordW = np.max([0,MinCoordW*SectionSize - RL ])
        MaxCoordH = np.min([pxH,MaxCoordH*SectionSize + SectionSize + RD])
        MaxCoordW = np.min([pxW,MaxCoordW*SectionSize + SectionSize + RR])
        
        print("selection for cluster ", cid , "from coordinates " , MinCoordW,",",  MinCoordH , "to ", MaxCoordW , ", " , MaxCoordH )
        
        ClusterImg = imgN[MinCoordH:MaxCoordH, MinCoordW: MaxCoordW]
        
        #print("Cluster", cid , "has size " , ClusterImg.shape[1],",",  ClusterImg.shape[0])
        
        # Image resizing to the specified Height and Width
        CIH = MaxCoordH - MinCoordH
        CIW = MaxCoordW - MinCoordW
        
 
        
        # Resize by filling the smallest dimension with empty pixels, to avoid stretching the img. 
        # Always put empty pixels above or to the right, so that origin of the new img is MinCoordW,MinCoordH
        
        #cv2.imshow('raw cutout img', ClusterImg)
        
        if CIH < CIW : # Need to fill in H direction before resizing
            # Target image ratio based on the aspect ratio desired
            TIW = CIW.astype(int)
            TIH = (TIW/AspectRatio).astype(int)
            
            diff = TIH-CIH
            newimg = np.zeros((TIH,TIW))
            if MinCoordH == 0: ### FILL DOWN
                newimg[0:(TIH-diff), 0:TIW] = ClusterImg
                MaxCoordH += diff
            else: ### FILL UP
                newimg[diff:TIH, 0:TIW] = ClusterImg
            ClusterImg = newimg
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        elif CIW < CIH : # need to fill in the W direction
            # Target image ratio based on the aspect ratio desired
            TIH = CIH.astype(int)
            TIW = (AspectRatio*TIH).astype(int)
            
            diff = TIW-CIW
            newimg = np.zeros((TIH,TIW))
            if MaxCoordW == CellsW -1: ### FILL LEFT
                newimg[0:TIH, (diff):TIW] = ClusterImg
                MinCoordW +-diff
            else: ### FILL RIGHT
                newimg[0:TIH, 0:CIW] = ClusterImg
            ClusterImg = newimg
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        else:
            ClusterImg = cv2.resize(ClusterImg, (TargetImgW, TargetImgH))
        
        
        #HC contains H and W coordinates from a top-left origin frame
        HeaderCoordinates[0,0] = int(MaxCoordH)
        HeaderCoordinates[1,0] = int(MinCoordW)   
        ExportImage[0,:,:] = ClusterImg
        
        namestring = "Cluster Image " + str(0)
        cv2.imshow(namestring , ExportImage[0,:,:].astype(np.uint8))
        
        
    end = time.time()
    print("total runtime was " , end-start, "seconds")
    print("Header coordinate H: ", HeaderCoordinates[0,0], " and coordinate W: ", HeaderCoordinates[1,0])
    #%% Displaying all the visuals
    
    # need to prepare some of the matrixes to be suitable for the imshow function
    #BinarySelectionimg = BinarySelectionMap.astype(np.uint8)
#    H = int(HeaderCoordinates[0,0])
#    W = int(HeaderCoordinates[1,0])
    #img[H,W] = MaxGrayScale
    
    
    cv2.rectangle(img,(MinCoordW,MinCoordH),(MaxCoordW,MaxCoordH),(250,250,250),3)
    img = img.astype(np.uint8)
    #cv2.imshow('Original', cv2.resize(imgo, (WindowW,WindowH)))
    cv2.imshow('Normalized', cv2.resize(img, (WindowW,WindowH)))
    
    #cv2.imshow('Binary cell map', cv2.resize(BinarySelectionimg, (WindowW, WindowH)))    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return ExportImage,HeaderCoordinates

#%% GENERAL NOTES
# Solar panels are problematic! need to find the SP frequency band as well...

