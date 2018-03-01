# reading condenseNet

## computationally efficience can be achieved by:
- removing redundant connections
- using low-precision or quantized weights
- using more efficient network architectures

## a good network for mobile devices needs:
- fast parallelization during training
- compactness at test time

## redundancy in networks
- layer by layer connectivity pattern forces network to replicate features from earlier layers throughout the network
  - denseNet can partially alleviate this need by directly connect each layer with all layers before it
- however. denseNet introduces redundancies when early features are not needed in later layers. This phenomenon becomes stronger as the layers goes deeper.

## weights pruning
- independent weight pruning generally achieves a high degree of sparsity, however, it requires storing a large number of indices.
- filter-level weight pruning achieves a lower degree of sparsity, but the resulting networks are much more regular.

## depth-wise separable conv VS group conv
- mobileNet, shuffleNet and Neural Architecture Search Network uses depth-wise separable conv, which is not well supported?
- condenseNet uses group conv.
- from what I see, these two kinds of convs have multiple outputs and reduced computational cost since they doesn't simply connect all the input channels to one output. I think of them as sub-architecture level optimization techniques.
- and why didn't you use the 1d conv?

## learned group conv
- add something about filter to the cost function to make the norm/altitude of a filter small
- drop the connections between the input/output filters in the 1x1 conv(lk->4k) by the sum of the L1 norm of weight between one input filter and all output filters
- condense coefficient C, drop 1/C connections one time, drop C-1 times, at last every group conv has 1/C input filters.
- C-1 times x 1/C connections drop is better than 1 time x (C-1)/C connections drop
- implemented as group conv at test time
- group conv effeicient G, 1x1 conv is divided into G times group conv
- C=2/4/8 @ G=4, leads to similar test error/FLOPs ratio, and are higher than C=1.
## updates towards standard denseNet
- exponentially increasing growth rate: $k=2^{m-1}k_{0}$, where $k_{0}$ is the initial growth rate and $m$ is the index of the dense block.
  - as it is shown that deeper layers in a denseNet tend to rely on high-level features more than low-level features. So, strengthening short-rante connections.
  - This way of setting growth rate does not introduce any additional hyper-parameter.
  - This way increases the computational efficiency but may decrease the parameter efficiency. Thus this is a trade-off.
- fully dense connectivity: adding extra down-sampling layers for high resolution feature maps and the down-sampled feature maps is used as the input of lower resolution feature maps in concat way.
  - wait! what's the difference here? in what way is the denseNet implemented? Why do I also think of this fully dense connectivity?
- these two points lead to the improvement in the test error/FLOPs ratio.

## some highlights
- cos learning rate, sgdr: stochastic gradient descent with restarts
- NEW idea get: REUSE everything avaliable