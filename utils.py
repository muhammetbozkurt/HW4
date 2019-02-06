import numpy as np
import cv2
import scipy.stats as st

GRAD_LIMIT = 90

def gradients(image,whereX,whereY,win_size):
    h, w = image.shape
    yAxis = [[1,2,1],
            [0,0,0],
            [-1,-2,-1]]
    
    xAxis = np.transpose(yAxis)
    resX =[]
    resY=[]
    for y in range(whereY, whereY+win_size-2):
        for x in range(whereX, whereX+win_size-2):
            a = sum(sum(image[x:x+3,y:y+3] * xAxis))
            b = sum(sum(image[x:x+3,y:y+3] * yAxis))
            if abs(a) > GRAD_LIMIT: 
                resX.append(a)
            else:
                resX.append(0)
            if abs(b) > GRAD_LIMIT:
                resY.append(b)
            else:
                resY.append(0)
    return resX, resY

def harris(image,win_size = 9,threshold = 5000):
    h, w = image.shape
    corners=[]
    
    for i in range(0,w-win_size,win_size):
        for j in range(0,h-win_size,win_size-1):
            Ix, Iy = gradients(image,i,j,win_size-1)
            squareX = sum(np.multiply(Ix,Ix))
            squareY = sum(np.multiply(Iy,Iy))
            mulXY = sum(np.multiply(Iy,Ix))
            G =np.array([[squareX,mulXY],
                        [mulXY,squareY]])
            e1,e2 = np.linalg.eigvals(G)
            if (abs(int(e1)) > threshold and abs(int(e2)) > threshold):
                corners.append([i,j])
    return corners

def gaus_kernel(kernel_len, sigma=3):
    interval = (2*sigma+1.)/(kernel_len)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernel_len+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def convolve2D(data,kernel):
    """i wrote this func for second hw"""
    h, w = data.shape
    kh, kw = kernel.shape
    new_image = np.zeros((h-kh+1,w-kh+1),dtype=np.uint8)#because image dimensions are going to be change depending on kernel size
    total=0
    for i in range(w-kw):
        for j in range(h-kw):
            for a in range(kh):
                for b in range(kh):
                    new_image.itemset((j,i),new_image.item(j,i)+data.item(j+a,i+b)*kernel.item(a,b))# look not so güvenilir
    return new_image

def masking(img):
    image = np.array(img)
    image[image<52] = 0
    image[image != 0] =255
    image= cv2.erode(image,np.ones((15,15)))
    image= cv2.dilate(image,np.ones((15,15)))
    return image

def drawingDots(data,image):
	for coor in data:
	    for i in range(4):
	        for j in range(4):
	            """because data holds upper left coordinate we need to skew it """
	            cv2.circle(image,(coor[1]+i+4,coor[0]+j+4),1,(0,0,255),-1) #

def masking(img):
    image = np.array(img)
    image[image<52] = 0
    image[image != 0] =255
    image= cv2.erode(image,np.ones((70,70)))
    image= cv2.dilate(image,np.ones((50,50)))
    return image

def which_mean(node,mean1,mean2):
    dist1 = ((node[0]-mean1[0])**2+(node[1]-mean1[1])**2+(node[2]-mean1[2])**2)**.5
    dist2 = ((node[0]-mean2[0])**2+(node[1]-mean2[1])**2+(node[2]-mean2[2])**2)**.5
    if(dist1<=dist2):
        return 1
    return 2

def new_means(image,group1,group2):
    total = np.array([0,0,0])
    for coor in group1:
        total += image[coor[0],coor[1]]
    mean1 = total/len(group1)
    total = np.array([0,0,0])
    for coor in group2:
        total += image[coor[0],coor[1]]
    mean2 = total/len(group2)
    return mean1, mean2

def k_means(image,rep=7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = masking(gray_image)
    h,w = gray_image.shape
    
    mean1 = [200,250,200]	#arbitrary mean1
    mean2 = [40,40,40]		#arbitrary mean2
    
    for repeat in range(rep):
        group1 = []
        group2 = []

        for i in range(w):
            for j in range(h):
                if(mask[j,i]>5):
                    condition = which_mean(image[j,i],mean1,mean2)
                    if(condition == 1):
                        group1.append([j,i])
                    else:
                        group2.append([j,i])
        mean1,mean2 = new_means(image,group1,group2)
    return group1,group2

def draw_seg(image):
    g1,g2=k_means(image)
    if(len(g1)<len(g2)):
        for coor in g1:
            image[coor[0],coor[1]] =[255,0,0]
    else:
        for coor in g1:
            image[coor[0],coor[1]] =[255,0,0]

def harris_cop(image):
	I = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	ker = gaus_kernel(5)
	I = convolve2D(I,ker)
	a = harris(I,win_size=9,threshold =50000)
	a = np.array(a)
	drawingDots(a,image)
	return image

if __name__ == "__main__":
	image = cv2.imread("blocks.jpg")
	image = harris_cop(image)
	cv2.imshow("deneme",image)
	cv2.waitKey()
	cv2.destroyAllWindows()

	image = cv2.imread("mr.jpg")
	draw_seg(image)
	cv2.imshow("deneme",image)
	cv2.waitKey()
	cv2.destroyAllWindows()

def ConnectedComponents(B): 
	X,Y = B.shape 
	L = np.zeros([X,Y]) 
	n=0 
	for (y,x) in B: 
		if B[y, x] and L[y, x]==0: 
			label(x,y,n,B,L) 
			n = n + 1 
	return L

def label(x_start,y_start,n,B,L): 
	L[y_start, x_start]= n 
	for (y,x) in N[y_start, x_start]: #N represents neighbour func  
		if L[y, x]==0 and B[y, x]: 
			label(x,y,n,B,L)