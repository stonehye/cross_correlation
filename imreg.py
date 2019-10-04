import cv2
import numpy as np
from matplotlib import pyplot as plt

# load two images
img = cv2.imread('source.jpg', 0)
img2 = cv2.imread('target.jpg', 0)

# apply fast fourier transform (FFT) to two images
G_a = np.fft.fft2(img)
G_b = np.fft.fft2(img2)
# multiply two elements
conj_a = np.ma.conjugate(G_a)
R = G_b * conj_a
R /= np.absolute(R)
# apply inverse FFT, r: result
r = np.fft.ifft2(R).real

# show plot
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img2, cmap='gray')
plt.title('translated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(r, cmap='gray')
plt.title('output Image'), plt.xticks([]), plt.yticks([])
plt.show()


