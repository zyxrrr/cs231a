import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

# Displays the HoG features next to the original image
def show_hog(orig, w, figsize = (8,6)):
    w = np.tile(w, [1, 1, 3])
    w = np.pad(w, ((0,0), (0,0), (0,5)), 'constant', constant_values=0.0)

    #  # Make pictures of positive and negative weights
    pos = hog_picture(w)
    neg = hog_picture(-w)

    # Put pictures together and draw
    buff = 10
    if w.min() < 0.0:
        pos = np.pad(pos, (buff, buff), 'constant', constant_values=0.5)
        neg = np.pad(neg, (buff, buff), 'constant', constant_values=0.5)
        im = np.hstack([pos, neg])
    else:
        im = pos

    im[im < 0] = 0.0
    im[im > 1] = 1.0
    plt.figure(figsize = figsize)
    plt.subplot(121)
    plt.imshow(orig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(im, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# Make picture of positive HOG weights.
def hog_picture(w, bs = 20):
    # construct a "glyph" for each orientaion
    bim1 = np.zeros((bs, bs))
    bim1[:,int(round(bs/2.0))-1:int(round(bs/2.0))+1] = 1.0
    bim = np.zeros((bs, bs, 9))
    bim[:,:,0] = bim1
    for i in xrange(1,9):
        bim[:,:,i] = imrotate(bim1, -i * float(bs), 'nearest') / 255.0

    # make pictures of positive weights bs adding up weighted glyphs
    w[w < 0] = 0.0
    im = np.zeros((bs * w.shape[0], bs * w.shape[1]))
    for i in xrange(w.shape[0]):
        for j in xrange(w.shape[1]):
            for k in xrange(9):
                im[i * bs : (i+1) * bs, j * bs : (j+1) * bs] += bim[:,:,k] * w[i,j,k+18]

    scale = max(w.max(), -w.max()) + 1e-8
    im /= scale
    return im
