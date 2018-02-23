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
#1 CReLU must be used in an efficient way, maybe DONE
#2 how to make every pixel in 32x32 get information from the 16x16 pixel block from the origin image before they are downsampled to 16x16?
#1dConv and tf.nn.dilation2d is preferable to expand reception field
#3 summary need to be debloated for performance reason
#4 high lrnSpd is disabled due to the remove of BN, does BN really means so much?
#5 add some noise to CReLU?
#6 VALID Convs and FCs are convertable.
#7 labelSmoothing 0.9/0.1 enabled and loss is messed up(~60+), why?
#8 indirection monitoring of varibles, eg. moments

class ModelForTheWin(object):
  def __init__(self,image,label,optimizer,l2DecayRate,isTraining,usingGpu):
    self.image=image
    self.label=label
    self.optimizer=optimizer
    self.l2DeacyRate=l2DecayRate

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

    self.isTraining=isTraining

    self.extraTrainOps=[]

    self.relu=self.myElu
    self.unit=self.my13Unit
    self.conv=self.my1dConv
    self.model=self.myModRight

  #Add some noise instead of 0?
  def myCRelu(self,x):
    with tf.variable_scope('crelu'):
      x=tf.concat([x,tf.negative(x)],axis=self.channelIndex)
      x=tf.nn.relu(x,name='Crelu')
      return x

  def myRelu(self,x):
    with tf.variable_scope('relu'):
      x=tf.nn.relu(x)
      tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
      return x

  def myElu(self,x):
    with tf.variable_scope('elu'):
      x=tf.nn.elu(x)
      return x

  def myConv(self,x,filterSize,strideLen,outFilters):
    with tf.variable_scope('conv'):
      inFilters=x.get_shape().as_list()[self.channelIndex]
      n=filterSize*filterSize*inFilters
      kernel=tf.get_variable(
          'weight',[filterSize,filterSize,inFilters,outFilters],tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/n)),
          regularizer=tf.contrib.layers.l2_regularizer(self.l2DeacyRate))

      strides=[1,strideLen,strideLen,strideLen]
      strides[self.channelIndex]=1

      x=tf.nn.conv2d(x,kernel,strides,data_format=self.dataFormat,padding='SAME')
      tf.summary.histogram('activations', x)
      return x

  #
  def my1dConv(self,x,filterSize,strideLen,outFilters):
    with tf.variable_scope('1dConv'):
      inFilters=x.get_shape().as_list()[self.channelIndex]
#      midFilters=outFilters#the number of filters between two 1Dconv
      midFilters=max(inFilters,outFilters)#better be the larger one of in/outFilters

      #n is right based on he init
      nRow=filterSize*inFilters
      kernelRow=tf.get_variable(
          'rowWeight',[1,filterSize,inFilters,midFilters],tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/nRow)),
          regularizer=tf.contrib.layers.l2_regularizer(self.l2DeacyRate))
      stridesRow=[1,1,1,1]
      stridesRow[self.widthIndex]=strideLen
      x=tf.nn.conv2d(x,kernelRow,stridesRow,data_format=self.dataFormat,padding='SAME',name='convRow')
      tf.summary.histogram('rowActivations', x)

      nCol=filterSize*midFilters
      kernelCol=tf.get_variable(
          'colWeight',[filterSize,1,midFilters,outFilters],tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/nCol)),
          regularizer=tf.contrib.layers.l2_regularizer(self.l2DeacyRate))
      stridesCol=[1,1,1,1]
      stridesCol[self.heightIndex]=strideLen
      x=tf.nn.conv2d(x,kernelCol,stridesCol,data_format=self.dataFormat,padding='SAME',name='convCol')
      tf.summary.histogram('colActivations', x)
    return x

  #for NHWC or NCHW only
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
        #To adjust l2decay for the numerical stability of weight, and now with BN there's maybe also a demand for this
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

  def myConvBnRelu(self,x,filterSize,strideLen,outFilters):
    if self.relu == self.myCRelu:
      x=self.conv(x,filterSize,strideLen,outFilters/2)
    else:
      x=self.conv(x,filterSize,strideLen,outFilters)
    x=self.myBatchNorm(x)
    return self.relu(x)

  def myBottleneckUnit(self,x,strideLen,outFilters):
    with tf.variable_scope('1st1x1'):#better renaming for better readability
      x=self.myConvBnRelu(x,1,strideLen,outFilters/4)
    with tf.variable_scope('2nd3x3'):
      x=self.myConvBnRelu(x,3,1,outFilters/4)
    with tf.variable_scope('3rd1x1'):
      x=self.myConvBnRelu(x,1,1,outFilters)
    return x

  #convert nchw to nhwc + using api + convert back
  #or using concat???
  #2.8->1.0 wtf
  def myNextBottleneckUnit(self,x,strideLen,outFilters):
    cardinality=32
    with tf.variable_scope('1st1x1'):
      x=self.myConvBnRelu(x,1,strideLen,outFilters/2)
      xList=tf.split(x,cardinality,axis=self.channelIndex)
    with tf.variable_scope('2nd3x3'):
      for i in range(cardinality):
        with tf.variable_scope('cardinality%d' % i):
          xList[i]=self.myConvBnRelu(xList[i],3,1,outFilters/2/cardinality)
      x=tf.concat(xList,self.channelIndex)
    with tf.variable_scope('3rd1x1'):
      x=self.myConvBnRelu(x,1,1,outFilters)
    return x

  def myNextBottleneckUnit2(self,x,strideLen,outFilters):
    cardinality=32
    with tf.variable_scope('1st1x1'):
      x=self.myConvBnRelu(x,1,strideLen,outFilters/2)
      xList=tf.split(x,cardinality,axis=self.channelIndex)
    with tf.variable_scope('2nd3x3'):
      for i in range(cardinality):
        with tf.variable_scope('cardinality%d' % i):
          xList[i]=self.myConvBnRelu(x,3,1,outFilters/2/cardinality)
      x=tf.concat(xList,self.channelIndex)
    with tf.variable_scope('3rd1x1'):
      x=self.myConvBnRelu(x,1,1,outFilters)#here are 50 maps * 2(positive+negetive) not so good
    return x

  def my13Unit(self,x,strideLen,outFilters):
    with tf.variable_scope('1st1x1'):
      x=self.myConvBnRelu(x,1,1,outFilters/4)
    with tf.variable_scope('2nd3x3'):
      x=self.myConvBnRelu(x,3,strideLen,outFilters)
    return x

  def myBottleneckUnitRight(self,x,strideLen,outFilters,preActivation=False):
    if preActivation:
      with tf.variable_scope('activateBeforeOri'):
        x=self.myBatchNorm(x)
        x=self.myElu(x)
        ori=x
    else:
      with tf.variable_scope('realOri'):
        ori=x
        x=self.myBatchNorm(x)
        x=self.myElu(x)
    with tf.variable_scope('1st1x1'):
      x=self.myConv(x,1,strideLen,outFilters/4)
    with tf.variable_scope('2st3x3'):
      x=self.myBatchNorm(x)
      x=self.myElu(x)
      x=self.my1dConv(x,3,1,outFilters/4)
    with tf.variable_scope('3st1x1'):
      x=self.myBatchNorm(x)
      x=self.myElu(x)
      x=self.myConv(x,1,1,outFilters)
    inFilters=ori.get_shape().dims[self.channelIndex]
    if inFilters != outFilters:
      with tf.variable_scope('projection'):
        ori=self.myConv(ori,1,strideLen,outFilters)
    with tf.variable_scope('merge'):
      x=x+ori
    return x


  def myResidual(self,x,strideLen,outFilters):
    inFilters=x.get_shape().as_list()[self.channelIndex]
    oriX=x

    x=self.unit(x,strideLen,outFilters)
    if inFilters != outFilters:
      with tf.variable_scope('project'):
        if self.relu == self.myCRelu:
          oriX=self.conv(oriX,1,strideLen,outFilters/2)
          oriX=self.relu(oriX)
        else:
          oriX=self.conv(oriX,1,strideLen,outFilters)
    return x+oriX

  def myGlobalAvgPool(self,x):
    return tf.reduce_mean(x,axis=[self.heightIndex,self.widthIndex])

  def myFC(self, x, outDim):#Only used in the last of the network
    with tf.variable_scope('fc'):
#      x = tf.reshape(x, [self.hps.batch_size, 1])#[bacth,value]
      w = tf.get_variable(
          'weight', [x.get_shape()[1], outDim],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [outDim],
                          initializer=tf.constant_initializer(value=0.0))#not 0.1 because it doesn't pass through relu
      return tf.nn.xw_plus_b(x, w, b)

  def myL2Decay(self):#maybe deprecated
    l2Costs=[]
    for var in tf.trainable_variables():
      if var.op.name.find(r'conv_weight')>0:
        l2Costs.append(tf.nn.l2_loss(var))
    return tf.multiply(self.l2DeacyRate,tf.add_n(l2Costs))

  def myMod(self):
    classes=100

    with tf.variable_scope('16x32'):
      conv16x32=self.my1dConv(self.image,filterSize=3,strideLen=1,outFilters=16)
      bn16x32=self.myBatchNorm(conv16x32)
      crelu32x32=self.myCRelu(bn16x32)
    with tf.variable_scope('32x32'):
      conv32x32=self.my1dConv(crelu32x32,filterSize=1,strideLen=1,outFilters=32)
      bn32x32=self.myBatchNorm(conv32x32)
      crelu64x32=self.myCRelu(bn32x32)
    with tf.variable_scope('64x16'):
      conv64x16=self.my1dConv(crelu64x32,filterSize=3,strideLen=3,outFilters=64)
      bn64x16=self.myBatchNorm(conv64x16)
      crelu128x16=self.myCRelu(bn64x16)
    with tf.variable_scope('128x16'):
      conv128x16=self.my1dConv(crelu128x16,filterSize=1,strideLen=1,outFilters=128)
      bn128x16=self.myBatchNorm(conv128x16)
      crelu256x16=self.myCRelu(bn128x16)
    with tf.variable_scope('256x8'):
      conv256x8=self.my1dConv(crelu256x16,filterSize=3,strideLen=3,outFilters=256)
      bn256x8=self.myBatchNorm(conv256x8)
      crelu512x8=self.myCRelu(bn256x8)
    with tf.variable_scope('512x8'):
      conv512x8=self.my1dConv(crelu512x8,filterSize=1,strideLen=1,outFilters=512)
      bn512x8=self.myBatchNorm(conv512x8)
      crelu1024x8=self.myCRelu(bn512x8)
    with tf.variable_scope('finalConv'):
      conv100x8=self.my1dConv(crelu1024x8,filterSize=3,strideLen=1,outFilters=classes)
      bn100x8=self.myBatchNorm(conv100x8)
    with tf.variable_scope('globalAvg'):
      avg100=self.myGlobalAvgPool(bn100x8)
      #softmax requires numerical stability, not helpful to remove the last BN, diverge
      logits=tf.subtract(avg100,tf.reduce_mean(avg100))
    with tf.variable_scope('predictions'):
      self.predictions=tf.nn.softmax(logits)
    with tf.variable_scope('accuracy'):
      truth=tf.argmax(self.label,axis=1)
      predictions=tf.argmax(self.predictions,axis=1)
      self.accuracy=tf.reduce_mean(tf.to_float(tf.equal(predictions,truth)))
    with tf.variable_scope('cost'):
      self.cost=tf.losses.softmax_cross_entropy(onehot_labels=self.label,logits=logits,label_smoothing=0.0)
  def myModRight(self):
    strides=[1,2,2]
    initFilters=16
    filters=[128,256,512]
    preActivation=[True,False,False]
    residualUnits=3
    classes=100

    with tf.variable_scope('initConv3to16'):
      x=self.my1dConv(self.image,3,1,initFilters)
    with tf.variable_scope('unit1x0'):
      x=self.myBottleneckUnitRight(x,strides[0],filters[0],preActivation=preActivation[0])
    for i in six.moves.range(1,residualUnits):
      with tf.variable_scope('unit1x%d' % i):
        x=self.myBottleneckUnitRight(x,1,filters[0],preActivation=False)
    with tf.variable_scope('unit2x0'):
      x=self.myBottleneckUnitRight(x,strides[1],filters[1],preActivation=preActivation[1])
    for i in six.moves.range(1,residualUnits):
      with tf.variable_scope('unit2x%d' % i):
        x=self.myBottleneckUnitRight(x,1,filters[1],preActivation=False)
    with tf.variable_scope('unit3x0'):
      x=self.myBottleneckUnitRight(x,strides[2],filters[2],preActivation=preActivation[2])
    for i in six.moves.range(1,residualUnits):
      with tf.variable_scope('unit3x%d' % i):
        x=self.myBottleneckUnitRight(x,1,filters[2],preActivation=False)
    with tf.variable_scope('lastPreActivation'):
      x=self.myBatchNorm(x)
      x=self.myElu(x)
    with tf.variable_scope('globalAvg'):
      x=self.myGlobalAvgPool(x)
    with tf.variable_scope('lastFc'):
      logits=self.myFC(x,classes)
    with tf.variable_scope('predictions'):
      self.predictions=tf.nn.softmax(logits)
    with tf.variable_scope('accuracy'):
      truth=tf.argmax(self.label,axis=1)
      predictions=tf.argmax(self.predictions,axis=1)
      self.accuracy=tf.reduce_mean(tf.to_float(tf.equal(predictions,truth)))
    with tf.variable_scope('cost'):
      crossEntropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.label)
      self.cost=tf.reduce_mean(crossEntropy)

  def myModel(self):
    strides=[1,2,2]
    filters=[16,128,256,512]
    residualUnits=3
    classes=100
    ori=False

#    with tf.variable_scope('init3to%d' % filters[0]):
#      x=self.myConvBnRelu(self.image,3,1,filters[0])

    with tf.variable_scope('unit1x0'):
      x=self.myResidual(self.image,strides[0],filters[1])
    for i in range(1,residualUnits):
        with tf.variable_scope('unit1x%d' % i):
          x=self.myResidual(x,1,filters[1])

    with tf.variable_scope('unit2x0'):
      x=self.myResidual(x,strides[1],filters[2])
    for i in range(1,residualUnits):
        with tf.variable_scope('unit2x%d' % i):
          x=self.myResidual(x,1,filters[2])

    with tf.variable_scope('unit3x0'):
      x=self.myResidual(x,strides[2],filters[3])
    for i in range(1,residualUnits):
        with tf.variable_scope('unit3x%d' % i):
          x=self.myResidual(x,1,filters[3])

    if ori:
      with tf.variable_scope('globalAvg'):
        x=self.myGlobalAvgPool(x)
      with tf.variable_scope('fc'):
        logits=self.myFC(x,classes)
    else:
      with tf.variable_scope('finalConv'):
  #      x=self.myConvBnRelu(x,1,1,classes)
        #for CReLU case, don't use ConvBnRelu directly
        x=self.conv(x,1,1,classes)
        x=self.myBatchNorm(x)#using only conv will cause NaN loss, WHY? Because This ISN't the LAST step to take
      with tf.variable_scope('globalAvg'):
        x=self.myGlobalAvgPool(x)
      with tf.variable_scope('fc'):
        logits=self.myFC(x,classes)

    with tf.variable_scope('predictions'):
      self.predictions=tf.nn.softmax(logits)

    with tf.variable_scope('accuracy'):
      truth=tf.argmax(self.label,axis=1)
      predictions=tf.argmax(self.predictions,axis=1)
      self.accuracy=tf.reduce_mean(tf.to_float(tf.equal(predictions,truth)))

    with tf.variable_scope('cost'):
      crossEntropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.label)
      self.cost=tf.reduce_mean(crossEntropy)
      self.cost+=self.myL2Decay()

    tf.logging.debug('image after unit %s', x.get_shape())#tf.logging interesting

  def buildTrainOp(self):
#    trainableVariables=tf.trainable_variables()
#    self.grads=tf.gradients(self.cost,trainableVariables)

    #momentum and lrnRate both need adjustment
    if self.optimizer == 'sgd':
      optimizer=tf.train.GradientDescentOptimizer(self.lrnRate)
    elif self.optimizer == 'mom':
      optimizer=tf.train.MomentumOptimizer(self.lrnRate,0.9)
    elif self.optimizer == 'nag':
      optimizer=tf.train.MomentumOptimizer(self.lrnRate,0.99,use_nesterov=True)

    self.grads=optimizer.compute_gradients(self.cost)

    applyOps=optimizer.apply_gradients(
        self.grads,
        global_step=self.globalStep)

    trainOps=[applyOps]+self.extraTrainOps
    self.trainOps=tf.group(*trainOps)

  def activationSummary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                         tf.nn.zero_fraction(x))

  def setSummary(self):#How to get more variable? Or rather, what variable do you want to get?
    tf.summary.scalar('cost',self.cost)
    tf.summary.scalar('lrnRate',self.lrnRate)
    if self.isTraining:#for training and evaluating, accuracy is treated differently
      tf.summary.scalar('accuracy',self.accuracy)

      # Add histograms for trainable variables.
      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

      # Add histograms for gradients.
      for grad, var in self.grads:
        if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)

      #TODO: add summarys for filters and feature maps

#      tf.summary.image(tf.get_variable("init_3_to_16/x"),)

  def buildGraph(self):
    self.globalStep=tf.contrib.framework.get_or_create_global_step()#fuck! need a tf.int32 cast workaround!
    #O(t^-1) lrnRate schedule?
    self.lrnRate=tf.train.piecewise_constant(tf.cast(self.globalStep,tf.int32),[25000,30000,35000],[0.1,0.03,0.01,0.003])

    self.model()
    if self.isTraining:
      self.buildTrainOp()
    self.setSummary()
    self.summaries=tf.summary.merge_all()

