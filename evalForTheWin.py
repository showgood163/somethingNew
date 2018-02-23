#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:29:39 2017

@author: asteria
"""

import tensorflow as tf
import numpy as np
import cifarInput
import Model

import six
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataSet', 'cifar100', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('checkPointPath', '/tmp/resnet',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('evalDataPath', '/tmp/resnet/data/test*',
                           'Filepattern for evaluating data.')
tf.app.flags.DEFINE_string('evalSummaryPath', '/tmp/resnet/eval',
                           'Directory to keep evaluating summaries.')
tf.app.flags.DEFINE_boolean('usingGpu', False,
                            'whether this program using GPU.')
tf.app.flags.DEFINE_integer('batchSize',100,
                            'mini batch size during evaluating.')
tf.app.flags.DEFINE_integer('evalBatchCount',100,
                            '# of evaled mini batch every time.')
tf.app.flags.DEFINE_boolean('isTraining', False,
                            'whether this program is used for training.')
tf.app.flags.DEFINE_string('optimizer', 'mom',
                           'the optimizer used in this program.')
tf.app.flags.DEFINE_float('l2DecayRate', 0.0002,
                           'L2 Decay Rate used in this program.')
tf.app.flags.DEFINE_integer('residualUnits',3,
                            '# of residual units per level.')
tf.app.flags.DEFINE_integer('initFilters',128,
                            '# of filters at the beginning.')

def evaluate(dataSet,image,label,optimizer,l2DecayRate,isTraining,usingGpu,residualUnits,initFilters):
  model=Model.ModelForTheWin(dataSet,image,label,optimizer,l2DecayRate,isTraining,usingGpu,residualUnits,initFilters)
  model.buildGraph()

  saver=tf.train.Saver()
  summaryWriter=tf.summary.FileWriter(FLAGS.evalSummaryPath)

  sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  bestAccuracy=0.0
  ckptStateLast=None

  while True:
    try:
      ckptState=tf.train.get_checkpoint_state(FLAGS.checkPointPath)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if ckptState!=ckptStateLast:
      if not (ckptState and ckptState.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.checkPointPath)
        continue
      tf.logging.info('Loading checkpoint %s', ckptState.model_checkpoint_path)
      saver.restore(sess, ckptState.model_checkpoint_path)

      totalPrediction,correctPrediction=0,0
      for _ in six.moves.range(FLAGS.evalBatchCount):
        (summaries, loss, predictions, truth, trainStep) = sess.run(
              [model.summaries, model.cost, model.predictions,
               model.label, model.globalStep])
        truth=np.argmax(truth,axis=1)
        predictions=np.argmax(predictions,axis=1)
        correctPrediction+=np.sum(truth == predictions)
        totalPrediction+=predictions.shape[0]

      accuracy=1.0*correctPrediction/totalPrediction
      bestAccuracy=max(accuracy,bestAccuracy)

      precisionSumm = tf.Summary()
      precisionSumm.value.add(
          tag='accuracy', simple_value=accuracy)
      summaryWriter.add_summary(precisionSumm, trainStep)
      bestPrecisionSumm = tf.Summary()
      bestPrecisionSumm.value.add(
          tag='bestAccuracy', simple_value=bestAccuracy)
      summaryWriter.add_summary(bestPrecisionSumm, trainStep)
      summaryWriter.add_summary(summaries, trainStep)
      tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                      (loss, accuracy, bestAccuracy))
      summaryWriter.flush()

      ckptStateLast=ckptState
    time.sleep(1)

def main(_):
  with tf.device('/cpu:0'):
    image,label=cifarInput.BuildInput(FLAGS.dataSet,FLAGS.evalDataPath,FLAGS.batchSize,FLAGS.isTraining,FLAGS.usingGpu)
  if FLAGS.usingGpu:
    with tf.device('/gpu:0'):
      evaluate(FLAGS.dataSet,image,label,FLAGS.optimizer,FLAGS.l2DecayRate,FLAGS.isTraining,FLAGS.usingGpu,FLAGS.residualUnits,FLAGS.initFilters)
  else:
    with tf.device('/cpu:0'):
      evaluate(FLAGS.dataSet,image,label,FLAGS.optimizer,FLAGS.l2DecayRate,FLAGS.isTraining,FLAGS.usingGpu,FLAGS.residualUnits,FLAGS.initFilters)
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()