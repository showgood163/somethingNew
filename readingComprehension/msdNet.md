# Multi Scale denseNet towards the best reuse

Towards:
- anytime classification
- bugeted batch classification

Insights:
- A computationally expensive model is good at classification, but it's a waste to use it to classify some easily classified images.
- Since computation is never free, how do we choose between either wasting computational resources by applying an unnecessarily expensive model to easy image, or making mistakes by using an efficient model that fails to recognize difficult images?
- because of limited time and computational resources, how to early exit with CNN? CNN is inherently not able to early exit.
- close to FractalNet, neural fabrics
- Vertical connections for multi-scale features

two principles:
- generate and maintain coarse level features throughout the network
  - to introduce intermediate classifiers even at early layers
- to inter-connect the layers with dense connectivity
  - to eusnre that these classifiers do not interfere with each other

P.S.
- to easy to get satisfied? Game hunger? Maybe not.