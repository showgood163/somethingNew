from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')

class ResNet(object):
  def __init__(self,hps,images,labels,mode,num_gpus):
    self.hps=hps
    self._images=images
    self.labels=labels
    self.mode=mode
    self.num_gpus=num_gpus

    self._extra_train_ops=[]

  def build_graph(self):
    self.global_step=tf.contrib.framework.get_or_create_global_step()
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries=tf.summary.merge_all()

  def _stride_arr(self,stride):
    if self.num_gpus == 0:
      return [1,stride,stride,1]
    else:
      return [1,1,stride,stride]

  def _conv(self,name,x,filter_size,in_filters,out_filters,strides):
    with tf.variable_scope(name):
      n=filter_size*filter_size*out_filters
      kernel=tf.get_variable(
          'DW',[filter_size,filter_size,in_filters,out_filters],tf.float32,
          initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      if self.num_gpus == 0:
        return tf.nn.conv2d(x,kernel,strides,data_format='NHWC',padding='SAME')
      else:
        return tf.nn.conv2d(x,kernel,strides,data_format='NCHW',padding='SAME')

  def _relu(self,x,leakiness=0.0):
    return tf.where(tf.less(x,0.0),leakiness*x,x,name='LReLU')

  def _batch_norm(self,name,x):
    with tf.variable_scope(name):
      if self.num_gpus == 0:
        params_shape=[x.get_shape()[-1]]
      else:
        params_shape=[x.get_shape()[1]]

      beta=tf.get_variable(
          'beta',params_shape,tf.float32,
          initializer=tf.constant_initializer(0.0,tf.float32))
      gamma=tf.get_variable(
          'gamma',params_shape,tf.float32,
          initializer=tf.constant_initializer(1.0,tf.float32))

      if self.mode == 'train':
        if self.num_gpus == 0:
          [y,mean,variance]=tf.nn.fused_batch_norm(
              x,gamma,beta,data_format='NHWC',epsilon=0.001)
        else:
          [y,mean,variance]=tf.nn.fused_batch_norm(
              x,gamma,beta,data_format='NCHW',epsilon=0.001)

        moving_mean=tf.get_variable(
            'moving_mean',params_shape,tf.float32,
            initializer=tf.constant_initializer(0.0,tf.float32),
            trainable=False)
        moving_variance=tf.get_variable(
            'moving_variance',params_shape,tf.float32,
            initializer=tf.constant_initializer(1.0,tf.float32))

        self._extra_train_ops.append(
            moving_averages.assign_moving_average(moving_mean,mean,0.9))
        self._extra_train_ops.append(
            moving_averages.assign_moving_average(moving_variance,variance,0.9))

      else:
        mean=tf.get_variable(
            'moving_mean',params_shape,tf.float32,
            initializer=tf.constant_initializer(0.0,tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name,mean)
        tf.summary.histogram(variance.op.name,variance)
        if self.num_gpus == 0:
          [y,_,_]=tf.nn.fused_batch_norm(
              x,gamma,beta,mean=mean,variance=variance,epsilon=0.001,data_format='NHWC',is_training=False)
        else:
          [y,_,_]=tf.nn.fused_batch_norm(
              x,gamma,beta,mean=mean,variance=variance,epsilon=0.001,data_format='NCHW',is_training=False)
      return y

  def _bottleneck_residual(self,x,in_filter,out_filter,stride,activite_before_residual=False):
    if activite_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x=self._batch_norm('init_bn',x)
        x=self._relu(x,self.hps.relu_leakiness)
        orig_x=x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x=x
        x=self._batch_norm('init_bn',x)
        x=self._relu(x,self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x=self._conv('conv1',x,1,in_filter,out_filter/4,stride)

    with tf.variable_scope('sub2'):
      x=self._batch_norm('bn2',x)
      x=self._relu(x,self.hps.relu_leakiness)
      x=self._conv('conv2',x,3,out_filter/4,out_filter/4,[1,1,1,1])

    with tf.variable_scope('sub3'):
      x=self._batch_norm('bn3',x)
      x=self._relu(x,self.hps.relu_leakiness)
      x=self._conv('conv3',x,1,out_filter/4,out_filter,[1,1,1,1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x=self._conv('project',orig_x,1,in_filter,out_filter,stride)
      x+=orig_x

    tf.logging.info('image after uynit %s', x.get_shape())
    return x

  def _my_residual(self,x,in_filter,out_filter,stride,activite_before_residual=False):
    if activite_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x=self._batch_norm('init_bn',x)
        x=self._relu(x,self.hps.relu_leakiness)
        orig_x=x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x=x
        x=self._batch_norm('init_bn',x)
        x=self._relu(x,self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x=self._conv('conv1',x,1,in_filter,out_filter/4,stride)

    with tf.variable_scope('sub2'):
      x=self._batch_norm('bn2',x)
      x=self._relu(x,self.hps.relu_leakiness)
      x=self._conv('conv2',x,3,out_filter/4,out_filter/4,[1,1,1,1])

    with tf.variable_scope('sub3'):
      x=self._batch_norm('bn3',x)
      x=self._relu(x,self.hps.relu_leakiness)
      x=self._conv('conv3',x,1,out_filter/4,out_filter,[1,1,1,1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x=self._conv('project',orig_x,1,in_filter,out_filter,stride)
      x+=orig_x

    tf.logging.info('image after uynit %s', x.get_shape())
    return x

  def _global_avg_pool(self,x):
    assert x.get_shape().ndims==4
    if self.num_gpus == 0:
      return tf.reduce_mean(x,[1,2])
    else:
      return tf.reduce_mean(x,[2,3])

  def _fully_connected(self,x,out_dim):
    x=tf.reshape(x,[self.hps.batch_size, -1])
    w=tf.get_variable(
        'DW',[x.get_shape()[1],out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b=tf.get_variable('biases',[out_dim],
                      initializer=tf.constant_initializer(value=0.0))
    return tf.nn.xw_plus_b(x,w,b)

  def _decay(self):
    costs=[]
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW')>0:
        costs.append(tf.nn.l2_loss(var))
    return tf.multiply(self.hps.weight_decay_rate,tf.add_n(costs))

  def _build_model(self):
    strides=[1,2,2]
    activate_before_residual=[True,False,False]

    res_func=self._bottleneck_residual
    filters=[16,64,128,256]

    with tf.variable_scope('init'):
      x=self._images
      x=self._conv('init_conv',x,3,3,filters[0],self._stride_arr(1))

    with tf.variable_scope('unit_1_0'):
      x=res_func(x,filters[0],filters[1],self._stride_arr(strides[0]),
                 activate_before_residual[0])
    for i in six.moves.range(1,self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x=res_func(x,filters[1],filters[1],self._stride_arr(1),False)

    with tf.variable_scope('unit_2_0'):
      x=res_func(x,filters[1],filters[2],self._stride_arr(strides[1]),
                 activate_before_residual[2])
    for i in six.moves.range(1,self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x=res_func(x,filters[2],filters[2],self._stride_arr(1),False)

    with tf.variable_scope('unit_3_0'):
      x=res_func(x,filters[2],filters[3],self._stride_arr(strides[2]),
                 activate_before_residual[2])
    for i in six.moves.range(1,self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x=res_func(x,filters[3],filters[3],self._stride_arr(1),False)

    with tf.variable_scope('unit_last'):
      x=self._batch_norm('final_bn',x)
      x=self._relu(x,self.hps.relu_leakiness)
      x=self._global_avg_pool(x)

    with tf.variable_scope('logits'):
      logits=self._fully_connected(x,self.hps.num_classes)
      self.predictions=tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent=tf.nn.softmax_cross_entropy(logits=logits,labels=self.labels)
      self.cost=tf.reduce_mean(xent,name='cross_entropy')
      self.cost += self._decay()

    tf.summary.scalar('cost',self.cost)

  def _build_train_op(self):
    self.lrn_rate=tf.constant(self.hps.lrn_rate,tf.float32)
    tf.summary.scalar('lrn_rate',self.lrn_rate)

    trainable_variables=tf.trainable_variables()
    grads=tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer=tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer=tf.train.MomentumOptimizer(self.lrn_rate,0.9)
    elif self.hps.optimizer == 'nag':
      optimizer=tf.train.MomentumOptimizer(self.lrn_rate,0.9,use_nesterov=True)

    apply_op=optimizer.apply_gradients(
        zip(grads,trainable_variables),
        global_step=self.global_step,name='step')

    train_ops=[apply_op]+self._extra_train_ops
    self.train_op=tf.group(*train_ops)