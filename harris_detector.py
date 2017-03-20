import numpy as np
from PIL import Image
# import conv2D slow
import gaussian
from skimage.feature import peak_local_max
import cv2
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import signal 
from operator import itemgetter

def harris_detector(imagepath,sigma,k,N):

	win = 2*sigma +1
	winsmooth = 2*sigma +1
	img = Image.open(imagepath).convert('L')
	im = np.array(img)

	gx = gaussian.gaussx(win, sigma)
	gy = gaussian.gaussy(win, sigma)

	# Ix = conv2D.conv2D(im,gx)	slow
	# Iy = conv2D.conv2D(im,gy)	slow

	Ix = signal.convolve2d(im,gx)
	Iy = signal.convolve2d(im,gy)

	Ixx = Ix*Ix
	Iyy = Iy*Iy
	Ixy = Ix*Iy

	gsmooth= gaussian.gauss(winsmooth, sigma)

	# Wxx = conv2D.conv2D(Ixx,gsmooth)	slow
	# Wyy = conv2D.conv2D(Iyy,gsmooth)	slow
	# Wxy = conv2D.conv2D(Ixy,gsmooth)	slow

	Wxx =  signal.convolve2d(Ixx,gsmooth)
	Wyy =  signal.convolve2d(Iyy,gsmooth)
	Wxy =  signal.convolve2d(Ixy,gsmooth)

	imfxx = Image.fromarray(Wxx)
	imfyy = Image.fromarray(Wyy)
	imfxy = Image.fromarray(Wxy)
	# imfxx.show()
	# imfyy.show()
	# imfxy.show()

	detA = Wxx*Wyy - Wxy**2
	traceA = Wxx + Wyy
	H = detA - k*traceA**2
	# himg = Image.fromarray(H)
	# himg.show()

	f = peak_local_max(H, min_distance=8,num_peaks=N,indices=True)

	siz= len(f)

	# print(len(y))
	for k in range(siz):
		cv2.circle(im,(itemgetter(1)(f[k]),itemgetter(0)(f[k])), 4, (0,0,0), thickness=1, lineType=1, shift=0)
	imgf = Image.fromarray(im)

	imgf.show()
	return f

