#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:45:27 2017

@author: asteria
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
import six

#TODO:
#1 CReLU must be implemented in an efficient way, maybe DONE
#this is a inspiration, ll-init.
#2 how to make every pixel in 32x32 get information from the 16x16 pixel block from the origin image before they are downsampled to 16x16?
#1dConv and tf.nn.dilation2d is preferable to expand reception field
#At least I can make all input information useful for training
#3 summary need to be debloated for performance reason
#now there's a demand to watch the output of every layer,using tf.GraphKeys
#4 high lrnSpd is disabled due to the remove of BN, does BN really means so much?
#for now, yes. Looking for a way to remove BN in the future
#5 add some noise to CReLU?
#Now using Elu for information passing through
#6 VALID Convs and FCs are convertable.
#They're basically the same, so choose whatever you like based on the speed/performance.
#7 labelSmoothing 0.9/0.1 enabled and loss is messed up(~60+), why?
#Wrong formula has been used, now it's working properly.
#8 indirect monitoring of varibles, eg. moments and gradients
#maybe?
#9 for elu, 2.0 on the init weight is not quite right..
#Yes but there are SELU and other improved version so this become less meaningful
#10 go and use tf.GraphKeys!FUCK!LOTS OF BUGS!GO AWAY!
#If you want to use Graph Collection, you need to put variables on CPU EXPLICITLY(too much bug or I DONT KNOW THE PRINCIPLE BEHIIND THIS.)
#11 if you don't use the default loss optimization function, you need to add L2loss yourself.
#Maybe different coefficient for conv and fc. This IS ONE REASON to USE CONV ONLY.
#And how about GAMMA and BETA in BN?
#12 make contribution towards RESIDUAL LEARNING, cleaning up the data shortcut.
#13 L2Decay@10^-4 makes sense for this model.
#The regularization loss in both SVM/Softmax cases could in this biological view be interpreted as gradual forgetting,
#since it would have the effect of driving all synaptic weights ww towards zero after every parameter update.
#this may seems interesting but is this optimistic?
#14 BLOCK TYPE may need experiment.
#15 #params x2 or sth. to expand the size of network.......
#16 inplement CReLU to expand the size and using it with the suggestion in the paper
#17 HOW to properly expand the size of network?
#18 label smoothing is like svm loss function
#19 NAG is better than MOM? Yes, and there's no need of greater momentum(0.99).
#20 adaGrad->rmsProp
#               +     => adaM
#            momentum
#21 elu with BN is not good as elu ONLY?! NaN LOSS!
#22 PReLU leads to overfit, ReLU << ELU
#23 orthogonal init@75%. how to combine it with var?
#24 use 3/i+3 instead of 1/i+3+1 to avoid thinking too much?
#25 avgPool with random channel/area? Nice idea.
#26 Linear CReLU? Nice.
#27 Seems like ReLU is computed on CPU not GPU and slow down the training
#28 Try 1x1conv+CReLU, no this is not good as it seems.
class ModelForTheWin(object):
  def __init__(self,dataSet,image,label,optimizer,l2DecayRate,isTraining,usingGpu,residualUnits,initFilters,Elu,strideChange,decompConv,original,numBeforeRes,order):
    self.image=image
    self.label=label
    self.optimizer=optimizer
    self.l2DeacyRate=l2DecayRate
    self.isTraining=isTraining
    self.residualUnits=residualUnits
    self.initFilters=initFilters


    if dataSet=='cifar10':
      self.classes=10
    elif dataSet=='cifar100':
      self.classes=100
    if usingGpu:
      self.channelIndex=1
      self.heightIndex=2
      self.widthIndex=3
      self.dataFormat='NCHW'
    else:
      self.channelIndex=3
      self.heightIndex=1
      self.widthIndex=2
      self.dataFormat='NHWC'

    self.extraTrainOps=[]

    self.Elu=Elu
    self.strideChange=strideChange
    self.decompConv=decompConv
    self.original=original
    self.numBeforeRes=numBeforeRes
    self.order=order

    self.growthRate=8
    self.layersPerBlock=6
    self.downSampleTimes=2

  def myL2Decay(self):#Use it!
    l2Costs=[]
    for var in tf.trainable_variables():
      if var.op.name.find(r'eight')>0:#for weight
        l2Costs.append(tf.nn.l2_loss(var))
    return tf.multiply(self.l2DeacyRate,tf.add_n(l2Costs))

  def myRelu(self,x):
    with tf.variable_scope('relu'):
      x=tf.nn.relu(x)
#      dont need this so far
#      tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
      return x

  #Add some noise instead of 0?
  #Using ELU instead of ReLU, training speed slows down apparently
  def myConcat(self,x,name='Concat'):
    x=tf.concat(x,axis=self.channelIndex,name=name)
    return x
  def myCRelu(self,x):
    with tf.variable_scope('crelu'):
      x=self.myConcat([x,tf.negative(x)])
      x=tf.nn.relu(x,name='Crelu')
      return x
  def myNCRelu(self,x):
    with tf.variable_scope('ncrelu'):
      x1=tf.nn.relu(x)
      x2=tf.negative(tf.nn.relu(tf.negative(x)))
      x=self.myConcat([x1,x2],name='NCrelu')
      return x
  def myMixedCRelu(self,x):
    with tf.variable_scope('mixedCrelu'):
      x1=tf.nn.relu(x)
      x2=tf.nn.relu(tf.negative(x))
      cr=self.myConcat([x1,x2],name='Crelu')
      x2=tf.negative(x2)
      ncr=self.myConcat([x1,x2],name='NCrelu')
      return cr,ncr

  def myElu(self,x):
    with tf.variable_scope('elu'):
      x=tf.nn.elu(x)
      return x

  def mySElu(self,x):
    with tf.variable_scope('selu'):
      alpha = 1.6732632423543772848170429916717
      scale = 1.0507009873554804934193349852946
      return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

  def mySwish(self,x):
    with tf.variable_scope('swish'):
      x=tf.multiply(tf.sigmoid(x),x)
      return x

  #for 4D only, channel-shared not channel-wise
  #OutOfMemory->working dirtyly
  #overfit
  def myPRelu(self,x):
    with tf.variable_scope('prelu'):
#      shape=x.get_shape().as_list()[self.channelIndex]
      shape=[1]
      alpha=tf.get_variable(
          'alpha',shape=shape,dtype=tf.float32,
          initializer=tf.zeros_initializer())
      return tf.add(x,tf.multiply(tf.nn.relu(tf.negative(x)),alpha))

  #HxW->1x1 global avg pool
  def myGlobalAvgPool(self,x):
    return tf.reduce_mean(x,axis=[self.heightIndex,self.widthIndex])

  #normal avg pool
  #I feel strange about this stride variable
  #in fact filterSize and strideLen are the same, so this typo isn't revealed, and it's fixed now.
  def myAvgPool(self,x,filterSize,strideLen):
    kernel=[1,filterSize,filterSize,filterSize]
    kernel[self.channelIndex]=1
    stride=[1,strideLen,strideLen,strideLen]
    stride[self.channelIndex]=1
    return tf.nn.avg_pool(x,kernel,stride,padding='VALID',data_format=self.dataFormat)

  def myMaxPool(self,x,filterSize,strideLen):
    kernel=[1,filterSize,filterSize,filterSize]
    kernel[self.channelIndex]=1
    stride=[1,strideLen,strideLen,strideLen]
    stride[self.channelIndex]=1
    return tf.nn.max_pool(x,kernel,stride,padding='VALID',data_format=self.dataFormat)

  #for 2D Only, used in the last of the network, may use valid conv instead
  #using he init
  def myFC(self, x, outDim):
    inFilters=x.get_shape().as_list()[1]
    with tf.variable_scope('fc'):
      w = tf.get_variable(
          'weight', [inFilters, outDim],dtype=tf.float32,
          initializer=tf.initializers.variance_scaling(scale=1.0,distribution='uniform'))
      b = tf.get_variable('biases', [outDim],
                          initializer=tf.constant_initializer(value=0.0))#not 0.1 because it doesn't pass through relu
    return tf.nn.xw_plus_b(x, w, b)

  #Conv with CReLU, where is BN?
#  crelu(x)
#  conv(x+,w)
#  conv(x-,-w)
  def myCreConv(self,x,filterSize,strideLen,outFilters,rowOrCol='none',padding='SAME',useBias=False):
    inFilters=x.get_shape().as_list()[self.channelIndex]
    xP=tf.nn.relu(x)
    xN=tf.nn.relu(tf.negative(x))
    with tf.variable_scope('conv'):
      n=filterSize*filterSize*inFilters
      #TODO: variance?
      kernel=tf.get_variable(
          'weight',[filterSize,filterSize,inFilters,outFilters],tf.float32,
          initializer=tf.orthogonal_initializer())

      strides=[1,strideLen,strideLen,strideLen]
      strides[self.channelIndex]=1

      xP=tf.nn.conv2d(xP,kernel,strides,data_format=self.dataFormat,padding=padding)
      xN=tf.nn.conv2d(xN,kernel,strides,data_format=self.dataFormat,padding=padding)
      x=self.myConcat([xP,xN])
      tf.summary.histogram('activations', x)
    return x

  #combine 1d and 2d
  #TODO: change the weight here for specific kind of ReLU
  def myConv(self,x,filterSize,strideLen,outFilters,rowOrCol='none',padding='SAME',useBias=False,useOrth=False):
    inFilters=x.get_shape().as_list()[self.channelIndex]

    if useOrth:
      initializer=tf.orthogonal_initializer()
    else:
      #FAN IN
      if rowOrCol == 'none':
        n=filterSize*filterSize*inFilters
      else:
        n=filterSize*inFilters
      initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
    #[filter_height, filter_width, in_channels, out_channels]
    kernelSize=[filterSize,filterSize,inFilters,outFilters]
    if rowOrCol == 'row':
      kernelSize[0]=1
    elif rowOrCol == 'col':
      kernelSize[1]=1

    kernel=tf.get_variable(
            'weight',kernelSize,tf.float32,initializer=initializer)

    strides=[1,1,1,1]
    if rowOrCol == 'none':
      strides[self.widthIndex]=strideLen
      strides[self.heightIndex]=strideLen
    elif rowOrCol == 'row':
      strides[self.widthIndex]=strideLen
    elif rowOrCol == 'col':
      strides[self.heightIndex]=strideLen

    x=tf.nn.conv2d(x,kernel,strides,data_format=self.dataFormat,padding=padding)

    if useBias == True:
      bias=tf.get_variable(
          'bias',[outFilters],tf.float32,
          initializer=tf.zeros_initializer())
      x=tf.nn.bias_add(x,bias,data_format=self.dataFormat)
    tf.summary.histogram('activations', x)
    return x

  #for 4D tensor only
  def myBatchNorm(self,x):
    epsilion=0.001
    decay=0.9

    with tf.variable_scope('batchNorm'):
      paramsShape=x.get_shape().dims[self.channelIndex]
      gamma=tf.get_variable(
          'gamma',paramsShape,tf.float32,
          initializer=tf.constant_initializer(1.0,tf.float32))
      beta=tf.get_variable(
          'beta',paramsShape,tf.float32,
          initializer=tf.constant_initializer(0.0,tf.float32))

      if self.isTraining:
        [y,mean,variance]=tf.nn.fused_batch_norm(
            x,gamma,beta,data_format=self.dataFormat,epsilon=epsilion)

        movingMean=tf.get_variable(
            'movingMean',paramsShape,tf.float32,
            initializer=tf.constant_initializer(0.0,tf.float32),trainable=False)
        movingVariance=tf.get_variable(
            'movingVariance',paramsShape,tf.float32,
            initializer=tf.constant_initializer(1.0,tf.float32),trainable=False)
        #To adjust l2decay for the numerical stability of weight(?), and now with BN there's maybe also a demand for this
        tf.summary.histogram('movingMean',movingMean)
        tf.summary.histogram('movingVariance',movingVariance)

        self.extraTrainOps.append(
            moving_averages.assign_moving_average(movingMean,mean,decay))
        self.extraTrainOps.append(
            moving_averages.assign_moving_average(movingVariance,variance,decay))
      else:
        mean=tf.get_variable(
            'movingMean',paramsShape,tf.float32,
            initializer=tf.constant_initializer(0.0,tf.float32),trainable=False)
        variance=tf.get_variable(
            'movingVariance',paramsShape,tf.float32,
            initializer=tf.constant_initializer(1.0,tf.float32),trainable=False)

        [y,_,_]=tf.nn.fused_batch_norm(
            x,gamma,beta,mean=mean,variance=variance,epsilon=epsilion,data_format=self.dataFormat,is_training=self.isTraining)
    return y
  def denseLayer(self,x):
    with tf.variable_scope('1x1_bottleNeck'):
      bn=self.myBatchNorm(x)
      relu=self.myRelu(bn)
      conv=self.myConv(relu,1,1,4*self.growthRate)
    with tf.variable_scope('3x3'):
      bn=self.myBatchNorm(conv)
      relu=self.myRelu(bn)
      conv=self.myConv(relu,3,1,self.growthRate)
    return conv
  def denseBlock(self,x):
    concatFeatures=x
    for i in range(self.layersPerBlock):
      with tf.variable_scope('denseBlock%d' % i):
        layerFeatures=self.denseLayer(concatFeatures)
        concatFeatures=self.myConcat([concatFeatures,layerFeatures])
    return concatFeatures
  def denseBlockStack(self,x):
    features=x
    for i in range(self.downSampleTimes):
      with tf.variable_scope('featureSizelevel%d' % i):
        features=self.denseBlock(features)
        features=self.myAvgPool(features,2,2)
    with tf.variable_scope('featureSizelevel%d' % self.downSampleTimes):
      features=self.denseBlock(features)
    return features


  def myModForTest(self,Elu,strideChange,decompConv,original,numBeforeRes,order):
    #activation function
    if Elu:
      actFun=self.myElu
    else:
      actFun=self.myRelu
    #main body
    with tf.variable_scope('initConv'):
      # x=self.myConv(self.image,3,1,self.initFilters/2)
      # x=self.myCRelu(x)
      x=self.myConv(self.image,3,1,self.initFilters)
    #x=self.myBottleneckUnitStackForTest(x,actFun,strideChange,decompConv,original,numBeforeRes,order)
    x=self.denseBlockStack(x)
    with tf.variable_scope('lastPreActivation'):
      ori2=tf.reduce_mean(tf.square(x))
      tf.summary.scalar('originalBeforeLastPreActivation',ori2)
      x=self.myBatchNorm(x)
      x=actFun(x)
    with tf.variable_scope('globalAvg'):
      x=self.myGlobalAvgPool(x)
    with tf.variable_scope('lastFc'):
      logits=self.myFC(x,self.classes)
    with tf.variable_scope('predictions'):
      self.predictions=tf.nn.softmax(logits)
    with tf.variable_scope('accuracy'):
      truth=tf.argmax(self.label,axis=1)
      predictions=tf.argmax(self.predictions,axis=1)
      self.accuracy=tf.reduce_mean(tf.to_float(tf.equal(predictions,truth)))
    with tf.variable_scope('cost'):
      self.costOri=tf.losses.softmax_cross_entropy(onehot_labels=self.label,logits=logits)
      self.cost=self.costOri+self.myL2Decay()

  def buildTrainOp(self):
    #TODO: momentum and lrnRate both need adjustment
    if self.optimizer == 'sgd':
      optimizer=tf.train.GradientDescentOptimizer(self.lrnRate)
    elif self.optimizer == 'mom':
      optimizer=tf.train.MomentumOptimizer(self.lrnRate,0.9)
    elif self.optimizer == 'nag':
      optimizer=tf.train.MomentumOptimizer(self.lrnRate,0.9,use_nesterov=True)
    elif self.optimizer == 'adam':
      optimizer=tf.train.AdamOptimizer(self.lrnRate)

    self.grads=optimizer.compute_gradients(self.cost)

    applyOps=optimizer.apply_gradients(
        self.grads,
        global_step=self.globalStep)

    trainOps=[applyOps]+self.extraTrainOps#+[updateOps]
    self.trainOps=tf.group(*trainOps)

  def setSummary(self):#How to get more variable? Or rather, what variable do you want to get?
    tf.summary.scalar('lrnRate',self.lrnRate)
    tf.summary.scalar('costOri',self.costOri)#Fuck ya, this is used to see how good the weight decay is.
    tf.summary.scalar('cost',self.cost)
    tf.summary.scalar('accuracy',self.accuracy)

    # Add histograms for trainable variables and its gradients, affected by L2.
    for grad, var in self.grads:
      if grad is not None:
        tf.summary.histogram(var.op.name, var)
        tf.summary.histogram(var.op.name + '/gradients', grad)

    #TODO: add summarys for filters and feature maps
    self.activationSummaries(tf.get_collection(tf.GraphKeys.ACTIVATIONS))

  def buildGraph(self):
    #It's easy if you TRY???
    if not(self.isTraining):
      tf.get_variable_scope().reuse_variables()
#    self.globalStep=tf.contrib.framework.get_or_create_global_step()#fuck! need a tf.int32 cast workaround!
    #self.globalStep=tf.get_variable('global_step',shape=[],dtype=tf.int32,initializer=tf.zeros_initializer(),trainable=False)
    self.globalStep=tf.train.get_or_create_global_step()
    #self.every100Steps=tf.assert_equal(tf.mod(self.globalStep,tf.constant(100,dtype=tf.int64)),tf.constant(0))
    #O(t^-1) lrnRate schedule?
    self.lrnRate=tf.train.piecewise_constant(self.globalStep,[20000,30000],[0.1,0.01,0.001])
#    self.lrnRate=tf.get_variable('lrn_rate',shape=[],dtype=tf.float32,initializer=tf.constant_initializer(value=0.1),trainable=False)

    self.myModForTest(self.Elu,self.strideChange,self.decompConv,self.original,self.numBeforeRes,self.order)
    if self.isTraining:
      self.buildTrainOp()
      self.setSummary()
    self.summaries=tf.summary.merge_all()

  def activationSummary(self,x):
    tf.summary.histogram(x.op.name + '/activations', x)

  def activationSummaries(self,endpoints):
    with tf.name_scope('summaries'):
      for act in endpoints:
        self.activationSummary(act)
