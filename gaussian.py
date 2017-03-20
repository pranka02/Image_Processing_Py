import numpy as np

def gauss(win, sigma):
   
    x = np.arange(0, win, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = win // 2

    g=1/(2*np.pi*sigma**2)*np.exp((((x-x0)**2+(y-y0)**2))/2*sigma**2)
    return g

def gaussx(win, sigma):
   
    x = np.arange(0, win, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = win // 2

    gx=(x-x0)/(2*np.pi*sigma**4)*np.exp((((x-x0)**2+(y-y0)**2))/2*sigma**2)
    return gx

def gaussy(win, sigma):
   
    x = np.arange(0, win, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = win // 2

    gy=(y-y0)/(2*np.pi*sigma**4)*np.exp((((x-x0)**2+(y-y0)**2))/2*sigma**2)
    return gy


