#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:15:26 2017

@author: asteria
"""

import tensorflow as tf

def BuildInput(dataSet, dataPath, batchSize, isTraining , usingGpu):
  imageSize=32
  depth=3
  if dataSet == 'cifar10':
    labelOffset=0
    labelBytes=1
    numClasses=10
  elif dataSet == 'cifar100':
    labelOffset=1
    labelBytes=1
    numClasses=100
  else:
    raise ValueError('Not supported dataset %s', dataSet)

  imageBytes=depth*imageSize*imageSize
  recordBytes=labelOffset+labelBytes+imageBytes
  dataFiles=tf.gfile.Glob(dataPath)
  fileQueue=tf.train.string_input_producer(dataFiles,shuffle=True)

  reader=tf.FixedLengthRecordReader(record_bytes=recordBytes)
  _,value=reader.read(fileQueue)

  record=tf.decode_raw(value,tf.uint8)

  label=tf.cast(
      tf.strided_slice(record,[labelOffset],[labelOffset+labelBytes]),
      tf.int32)
  dataCHW=tf.cast(#for GPU
      tf.reshape(
          tf.strided_slice(record,[labelOffset+labelBytes],[recordBytes]),
          [depth,imageSize,imageSize]),
      tf.float32)
  dataHWC=tf.transpose(dataCHW,[1,2,0])#for image PreProcessing on CPU

  if isTraining:
    #PreProcessing
    image=tf.image.resize_image_with_crop_or_pad(
        dataHWC,imageSize+4,imageSize+4)
    image=tf.random_crop(image,[imageSize,imageSize,3])
    image=tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    numThreads=16
    if usingGpu:
      image=tf.transpose(image,[2,0,1])
      feedQueue=tf.RandomShuffleQueue(
          capacity=numThreads*batchSize,
          min_after_dequeue=8*batchSize,
          dtypes=[tf.float32,tf.int32],
          shapes=[[depth,imageSize,imageSize],[1]])
    else:
      feedQueue=tf.RandomShuffleQueue(
          capacity=numThreads*batchSize,
          min_after_dequeue=8*batchSize,
          dtypes=[tf.float32,tf.int32],
          shapes=[[imageSize,imageSize,depth],[1]])
  else:
    image = tf.image.per_image_standardization(dataHWC)

    numThreads=16
    if usingGpu:
      image=tf.transpose(image,[2,0,1])
      feedQueue=tf.FIFOQueue(
          capacity=numThreads*batchSize,
          dtypes=[tf.float32,tf.int32],
          shapes=[[depth,imageSize,imageSize],[1]])
    else:
      feedQueue=tf.FIFOQueue(
          capacity=numThreads*batchSize,
          dtypes=[tf.float32,tf.int32],
          shapes=[[imageSize,imageSize,depth],[1]])

  enqueueOp=feedQueue.enqueue([image,label])
  tf.train.add_queue_runner(
      tf.train.queue_runner.QueueRunner(
          feedQueue,[enqueueOp]*numThreads))

  images,labels=feedQueue.dequeue_many(batchSize)
  labels=tf.reshape(labels,[batchSize,1])#densed list
  indices=tf.reshape(tf.range(0,batchSize,1),[batchSize,1])
  labels=tf.sparse_to_dense(#it's easy. The name of 1st parameter is sparse_indices. The name of this func means Converts a sparse representation(index) into a dense tensor(matrix)
      tf.concat(values=[indices,labels],axis=1),#sparsed matrix
      [batchSize,numClasses],
      1.0,0.0)#for label smoothing, maybe with tf.losses.softmax_cross_entropy it's easier to implement
#              Nope! The formula for label smoothing is:
#              smooth_positives = 1.0 - label_smoothing
#              smooth_negatives = label_smoothing / num_classes
#              one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
#              dont change here, change in the loss function

  if usingGpu:
    tf.summary.image('processedImage',tf.transpose(images,[0,2,3,1]))
  else:
    tf.summary.image('processedImage',images)

  return images,labels