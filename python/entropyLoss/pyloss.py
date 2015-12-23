import caffe
import numpy as np

class EntropyCode(caffe.Layer):
   def setup(self, bottom, top):
      if len(bottom) != 1:
         raise Exception("Need one input to compute annealing.")
      if len(top) != 1:
         raise Exception("Need one output to compute annealing.")
      #self.beta_ = layer_params['beta']

   def reshape(self, bottom, top):
      botshape = bottom[0].shape
      # loss output is scalar
      top[0].reshape(botshape)

   def forward(self, bottom, top):
      top[0].data[...] = np.copy(bottom[0].data)
      #totsum = np.sum(np.exp(bottom[0].data))
      #top[0].data[...] = np.exp(bottom[0].data)/totsum * (bottom[0].data - np.log(totsum))

   def backward(self, top, propagate_down, bottom):
      #bottom[0].diff[...] = np.copy(top[0].diff)
      pass
