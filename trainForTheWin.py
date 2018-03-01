#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:09:04 2017

@author: asteria
"""

import tensorflow as tf
import sys
import cifarInput
import modelNew as Model
import six

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataSet', 'cifar100', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('checkPointPath', '/tmp/resnet',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('trainDataPath', '/tmp/cifar100/train*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('trainSummaryPath', '/tmp/resnet/train',
                           'Directory to keep training summaries.')
tf.app.flags.DEFINE_string('evalDataPath', '/tmp/cifar100/test*',
                           'Filepattern for evaluating data.')
tf.app.flags.DEFINE_string('evalSummaryPath', '/tmp/resnet/eval',
                           'Directory to keep evaluating summaries.')
tf.app.flags.DEFINE_boolean('usingGpu', True,
                            'whether training using GPU.')
tf.app.flags.DEFINE_integer('batchSize',64,
                            'mini batch size during training.')
tf.app.flags.DEFINE_integer('evalBatchSize',100,
                            'mini batch size during evaluating.')
tf.app.flags.DEFINE_integer('evalBatchCount',100,
                            '# of evaled mini batch every time.')
tf.app.flags.DEFINE_integer('totalStep',40000,
                            '# of steps to train a model.')
tf.app.flags.DEFINE_string('optimizer', 'nag',
                           'the optimizer used in this program.')
tf.app.flags.DEFINE_float('l2DecayRate', 0.0001,
                           'L2 Decay Rate used in this program.')
tf.app.flags.DEFINE_integer('residualUnits',3,
                            '# of residual units per level.')
tf.app.flags.DEFINE_integer('initFilters',32,
                            '# of filters at the beginning.')

tf.app.flags.DEFINE_boolean('Elu', False,
                            'whether useing ELU activation function in the training.')
tf.app.flags.DEFINE_boolean('strideChange', True,
                            'whether changing the stride location in the Residual Unit/Block.')
tf.app.flags.DEFINE_boolean('decompConv', False,
                            'whether using 2x1D convolution instead of 1x2D one.')
tf.app.flags.DEFINE_boolean('original', False,
                            'whether this model is original.')
tf.app.flags.DEFINE_integer('numBeforeRes',1,
                            'how many of avgP and CReLU is placed before the Residual Function.')
tf.app.flags.DEFINE_boolean('order', True,
                            'True: CReLU is placed before avgP; False avgP is placed before CReLU.')

tf.app.flags.DEFINE_boolean('trainWithEval', True,
                            'whether evaluating during training.')




def train(dataSet,imageTrain,labelTrain,imageEval,labelEval,optimizer,l2DecayRate,usingGpu,residualUnits,initFilters,Elu,strideChange,decompConv,original,numBeforeRes,order,trainWithEval):
#  def __init__(self,dataSet,image,label,optimizer,l2DecayRate,isTraining,usingGpu,residualUnits,initFilters,Elu,strideChange,decompConv,original,numBeforeRes,order):
  modelTrain=Model.ModelForTheWin(dataSet,imageTrain,labelTrain,optimizer,l2DecayRate,True,usingGpu,residualUnits,initFilters,Elu,strideChange,decompConv,original,numBeforeRes,order)
  modelTrain.buildGraph()

  if trainWithEval:
    modelEval=Model.ModelForTheWin(dataSet,imageEval,labelEval,optimizer,l2DecayRate,False,usingGpu,residualUnits,initFilters,Elu,strideChange,decompConv,original,numBeforeRes,order)
    modelEval.buildGraph()
    bestAccuracy=0.0
    evalSummaryWriter=tf.summary.FileWriter(FLAGS.evalSummaryPath)

  param_stats = tf.profiler.profile(
      tf.get_default_graph(),
      options=tf.profiler.ProfileOptionBuilder
          .trainable_variables_parameter())
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())

  loggingHook=tf.train.LoggingTensorHook(
      tensors={'step':modelTrain.globalStep,
               'loss':modelTrain.cost,
               'accuracy':modelTrain.accuracy,
               'lrnRate':modelTrain.lrnRate},
      every_n_iter=50)

  summaryHook=tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.trainSummaryPath,
      summary_op=modelTrain.summaries)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkPointPath,
    #   hooks=[loggingHook,
    #          summaryHook,
    #          tf.train.NanTensorHook(modelTrain.cost),
    #          tf.train.StopAtStepHook(last_step=FLAGS.totalStep+2)],#not loopable? need to be put in []
    #   save_checkpoint_secs=10,
    #   save_summaries_steps=None,
      config=tf.ConfigProto(allow_soft_placement=True)) as sess:#,log_device_placement=True
    #get current steps first, type int
    trainStep= tf.train.global_step(sess,modelTrain.globalStep)
    #train & eval, here using python int type to control
    for step in range(trainStep,FLAGS.totalStep+1):
      #eval and summary writing, omit step 0 because of 70+ loss........
      if trainWithEval and ((step != 0 and step%100 == 0) or step == 1):
        #evaluating!
        (cost, costOri, accuracy, trainStep) = sess.run(
                [modelEval.cost, modelEval.costOri,
                 modelEval.accuracy,modelEval.globalStep])
        totalAcc=accuracy
        totalCost=cost
        totalCostOri=costOri
#        tf.logging.info('EVAL!cost: %.3f, costOri: %.3f, accuracy: %.3f' % (cost, costOri, accuracy))
        for i in six.moves.range(FLAGS.evalBatchCount-1):
          (cost, costOri, accuracy) = sess.run(
                [modelEval.cost, modelEval.costOri, modelEval.accuracy])
          totalAcc+=accuracy
          totalCost+=cost
          totalCostOri+=costOri
#          tf.logging.info('EVAL!cost: %.3f, costOri: %.3f, accuracy: %.3f' % (cost, costOri, accuracy))

        accuracy=totalAcc/FLAGS.evalBatchCount
        cost=totalCost/FLAGS.evalBatchCount
        costOri=totalCostOri/FLAGS.evalBatchCount
        bestAccuracy=max(accuracy,bestAccuracy)
        tf.logging.info('EVALstep: %d, cost: %.3f, costOri: %.3f, accuracy: %.3f,  best accuracy: %.3f' %
                        (trainStep, cost, costOri, accuracy, bestAccuracy))
        #eval summary writing
        costSumm = tf.Summary()
        costSumm.value.add(
            tag='cost', simple_value=cost)
        evalSummaryWriter.add_summary(costSumm, trainStep)
        costOriSumm = tf.Summary()
        costOriSumm.value.add(
            tag='costOri', simple_value=costOri)
        evalSummaryWriter.add_summary(costOriSumm, trainStep)
        accuracySumm = tf.Summary()
        accuracySumm.value.add(
            tag='accuracy', simple_value=accuracy)
        evalSummaryWriter.add_summary(accuracySumm, trainStep)
        bestAccuracySumm = tf.Summary()
        bestAccuracySumm.value.add(
            tag='bestAccuracy', simple_value=bestAccuracy)
        evalSummaryWriter.add_summary(bestAccuracySumm, trainStep)
        evalSummaryWriter.flush()

      #training
      sess.run(modelTrain.trainOps)

def main(_):
  with tf.device('/cpu:0'):
    imageTrain,labelTrain=cifarInput.BuildInput(FLAGS.dataSet,FLAGS.trainDataPath,FLAGS.batchSize,True,FLAGS.usingGpu)
    imageEval,labelEval=cifarInput.BuildInput(FLAGS.dataSet,FLAGS.evalDataPath,FLAGS.evalBatchSize,False,FLAGS.usingGpu)
  if FLAGS.usingGpu:
    with tf.device('/gpu:0'):
      train(FLAGS.dataSet,imageTrain,labelTrain,imageEval,labelEval,FLAGS.optimizer,FLAGS.l2DecayRate,FLAGS.usingGpu,FLAGS.residualUnits,FLAGS.initFilters,FLAGS.Elu,FLAGS.strideChange,FLAGS.decompConv,FLAGS.original,FLAGS.numBeforeRes,FLAGS.order,FLAGS.trainWithEval)
  else:
    with tf.device('/cpu:0'):
      train(FLAGS.dataSet,imageTrain,labelTrain,imageEval,labelEval,FLAGS.optimizer,FLAGS.l2DecayRate,FLAGS.usingGpu,FLAGS.residualUnits,FLAGS.initFilters,FLAGS.Elu,FLAGS.strideChange,FLAGS.decompConv,FLAGS.original,FLAGS.numBeforeRes,FLAGS.order,FLAGS.trainWithEval)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()