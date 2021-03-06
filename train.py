#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:41:48 2017

@author: asteria
"""

import time
import six
import sys

import cifarInput
import numpy as np
import resnetModel
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar100', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('train_data_path', '/tmp/resnet/data/data*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '/tmp/resnet/data/test*',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/tmp/resnet/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 100,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/tmp/resnet',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')

#train or eval
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
#using cpu or gpu
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')

def train(hps,images,labels):
  model=resnetModel.ResNet(hps,images,labels,FLAGS.mode,FLAGS.num_gpus)
  model.build_graph()

  param_stats=tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total params: %d' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth=tf.argmax(model.labels,axis=1)
  predictions=tf.argmax(model.predictions,axis=1)
  precision=tf.reduce_mean(tf.to_float(tf.equal(predictions,truth)))

  summary_hook=tf.train.SummarySaverHook(#not good,need change
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge(
          [model.summaries,tf.summary.scalar('precision',precision)]))

  logging_hook=tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision,
               'lrnRate': model.lrn_rate},
      every_n_iter=100)

  class _LrnRateSetHook(tf.train.SessionRunHook):
    def begin(self):
      self._lrn_rate=0.3
    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,
          feed_dict={model.lrn_rate:self._lrn_rate})
    def after_run(self,run_context,run_values):
      train_step=run_values.results
      if train_step < 8000:
        self._lrn_rate = 0.3
      elif train_step < 12000:
        self._lrn_rate = 0.1
      elif train_step < 14000:
        self._lrn_rate = 0.03
      elif train_step < 15000:
        self._lrn_rate = 0.01
      elif train_step < 15500:
        self._lrn_rate = 0.003
      elif train_step < 15750:
        self._lrn_rate = 0.001
      elif train_step < 15875:
        self._lrn_rate = 0.0003
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook,_LrnRateSetHook()],
      chief_only_hooks=[summary_hook],
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as monSess:
    while not monSess.should_stop():
      monSess.run(model.train_op)

def evaluate(hps,images,labels):
  model=resnetModel.ResNet(hps,images,labels,FLAGS.mode,FLAGS.num_gpus)
  model.build_graph()
  saver=tf.train.Saver()
  summary_writer=tf.summary.FileWriter(FLAGS.eval_dir)

  sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision=0.9
  ckpt_state_last=tf.train.get_checkpoint_state(FLAGS.log_root)
  while True:
    try:
      ckpt_state=tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if FLAGS.eval_once or ckpt_state!=ckpt_state_last:
      if not(ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      saver.restore(sess,ckpt_state.model_checkpoint_path)

    total_prediction,correct_prediction=0,0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries,loss,predictions,truth,train_step)=sess.run(
          [model.summaries,model.cost,model.predictions,model.labels,model.global_step])
      truth=np.argmax(truth,axis=1)
      predictions=np.argmax(predictions,axis=1)
      correct_prediction+=np.sum(truth==predictions)
      total_prediction+=predictions.shape[0]

    precision=1.0*correct_prediction/total_prediction
    best_precision=max(precision,best_precision)

    precision_summ=tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ,train_step)

    precision_summ=tf.Summary()
    precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(precision_summ,train_step)

    summary_writer.add_summary(summaries,train_step)

    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                      (loss, precision, best_precision))
    summary_writer.flush()

    ckpt_state_last=ckpt_state

    if FLAGS.eval_once:
      break

    time.sleep(1)

def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnetModel.HParams(
      batch_size=batch_size,
      num_classes=num_classes,
      min_lrn_rate=0.0001,
      lrn_rate=0.1,
      num_residual_units=5,
      use_bottleneck=True,
      weight_decay_rate=0.0,#0.0002
      relu_leakiness=0.1,
      optimizer='nag')

  if FLAGS.mode == 'train':
    with tf.device('/cpu:0'):
      images, labels = cifarInput.BuildInput(
        FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode, FLAGS.num_gpus)
    with tf.device(dev):
        train(hps, images, labels)
  elif FLAGS.mode == 'eval':
    with tf.device('/cpu:0'):
      images, labels = cifarInput.BuildInput(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode, FLAGS.num_gpus)
    with tf.device(dev):
        evaluate(hps, images, labels)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
