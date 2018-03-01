# Notes on DenseNet 

## The skip conncetion used in the ResNet is good for the gradient to flow through the whole network. However, the `add` way of merging results may impede the information flow in the network.

## To improve the information flow between layers, concatenating the result of layers instead of adding, which will result in:
- decrease in both computational cost and \#params since concatenate greatly decreases the output \#channels of convolutions in the whole network
- more feature reuses
- less overfitting

## the convolution micro-architecture: `BN`->`ReLU`->`Conv`

## down-sampling: `1x1 Conv`-> `2x2 avgPool`

## `growth rate` is the output \#channels of every layer. Because the concatenation, each layer can easily access to the information of every preceding layer, denoted as `k`

## `bottleneck layers`: `BN`->`ReLU`->`1x1 Conv@4k`->`BN`->`ReLU`->`3x3 Conv@k`. Computationally efficient, denoted as `B`

## `compression`:  decreasing the \#channels in the down-sampling process, denoted as `C`, normally the coefficient is 0.5.

## implementation details
- \#channels starting at 16 or 2xGrowth rate(BC)
- learning rate: 0.1->0.01@50%->0.001@75%
- weight decay: 0.0001
- learning algorithm: nag@0.9
- dropout@0.2 added after each conv layer except the initial conv layer for models without data augmentation, meaning that down-sampling layers are used with dropout.
- If there're data segmentation, then no need for dropout.

## the weight of every layer is directly connected to the loss function if down-sampling 1x1 conv is not considered. This means great gradient back propogation condition. The Deeply-supervised nets uses classifiers attached to every hidden layer to enforce the intermediate layers to learn discriminative features. DenseNet does it implicitly.


# P.S.
- Search masters' recent activities to get updated with information. Go for their blog, github or other online media.
- It is really helpful when you translate the English paper into Chinese. It helps in many ways, so here's the question: how to install a nice Chinese input method?
- Using crelu with dropout to work with the \#channel down? Not ok because dropout doesn't really decreases \#channels. If so, how to decrease \#channels?