import caffe
import numpy as np
import matplotlib.pyplot as plt
import IPython
import sys

sys.path.insert(0,"/Users/slundquist/workspace/caffe/python/entropyLoss/")

basedir = "/Users/slundquist/workspace/caffe/"

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return np.squeeze(data)


#net = caffe.Net("caffenet.prototxt", caffe.TRAIN)
solver = caffe.SGDSolver(basedir + "/models/entropy/solver.prototxt")
net = solver.net
#solver.net.copy_from(basedir + "models/entropy/output/lenet_iter_10000.caffemodel")

solver.solve()

img = vis_square(net.params['entConv1'][0].data.transpose(0, 2, 3, 1))
plt.imshow(img, cmap="Greys", interpolation="nearest")
plt.show(block=False)

IPython.embed()



