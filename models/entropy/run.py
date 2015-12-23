import caffe
import IPython
import sys

sys.path.insert(0,"/Users/slundquist/workspace/caffe/python/entropyLoss/")


#net = caffe.Net("caffenet.prototxt", caffe.TRAIN)
solver = caffe.SGDSolver("solver.prototxt")
net = solver.net
IPython.embed()

