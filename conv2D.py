# This program implements a 2D convolution function.

import numpy as np
import cv2


def conv2D(inp,ker):
	ir = len(inp)												# no of rows in input 
	ic = len(inp[0])											# no of columns in input 


	kr = len(ker)												# no of rows in kernel
	kc = len(ker[0])											# no of columns in kernel

	m = ir + kr -1												# no of rows in output
	n = ic + kc -1												# no of columns in output 

	y = np.zeros((m,n))

	kcentx =round(kr/2- 0.5)									# x coordinate of center of kernel 
	kcenty =round(kc/2- 0.5)									# y coordinate of center of kernel 

	padrow = round(kr/2)
	padcol = round(kc/2)
	inp = np.lib.pad(inp,(padrow,padcol),'constant',constant_values= 0)	# zero-padding input 

	inr = len(inp)												# no of rows in modified input
	inc = len(inp[0])											# no of columns in modified input

	length = kr -kcenty											# length from center of kernel
	width = kc -kcentx											# width from center of kernel
	

	rowmin = length
	rowmax = length
	colmin = width
	colmax = width

	ker = np.flip(ker, axis=0)									# flipping kernel horizontally and vertically
	ker = np.fliplr(ker)

	for k in range(m):											# convolution starts
			
		for l in range(n):
			
			y[k][l] = 0
			s = k-rowmin
			t = k+rowmax
			u = l-colmin
			v = l+colmax

			if s<0:
				s=max(0,s)
			elif s>=0:
				s +=1


			if t>m:
				t=min(m,t)

			if u<0:
				u=max(0,u)
			elif u>=0:
				 u+=1

			if v>m:
				v=min(n,v)


			for i in range(s,t):
				
				for j in range(u,v):
					p = i-k+kcentx
					q = j-l+kcenty
					y[k][l] += inp[i][j]*ker[p][q]

	return y
		
	




