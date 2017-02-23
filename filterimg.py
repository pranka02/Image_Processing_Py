import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array
from conv2D import conv2D

def filtimg(imagepath,filt):
	img = Image.open(imagepath).convert('L')					# open image and convert to grayscale
	inp = array(img)
	lenimg= len(inp)
	widimg = len(inp[0])
	size = lenimg*widimg							        # convert to numpy array
	x = inp.dtype
	img.show()									# display image 
							
	y = conv2D(inp,filt)								
	siz1= len(y)


	for i in range(siz1):								# normalizing pixel values  
		for j in range(siz1):
			if y[i][j] >255:
				y[i][j] =255
			if y[i][j] <0:
				y[i][j] =0

	y = y.astype(np.uint8)								# converting to unit8

	fftimg = np.fft.fft2(inp)
	fftimgs =np.fft.fftshift(fftimg)									# calculating fft of input image 
	fftfiltimg = np.fft.fft2(y)
	fftfiltimgs =np.fft.fftshift(fftfiltimg)							        # calculating fft of filtered image 
	fftfilt = np.fft.fft2(filt, s=(lenimg, widimg))				
	fftfilts = np.fft.fftshift(fftfilt)

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
	ax[0].imshow(abs(np.log(fftimgs)), interpolation="none", origin='lower')
	ax[0].set_title('log magnitude of input image')
	ax[1].imshow(abs(np.log(fftfiltimgs)), interpolation="none", origin='lower')
	ax[1].set_title('log magnitude of filtered image')
	ax[2].imshow(abs((fftfilts)), interpolation="none", origin='lower')
	# ax[2].imshow(fftfilt.ravel(), bins=100) take teh magnitude of the fft before histogram
	ax[2].set_title('frequency response of filter')
	plt.savefig('plot.png')
	plt.show()

	filtimg= Image.fromarray(y, 'L')
	filtimg.show()
	filtimg.save('filtimage.png')
	
	return 



