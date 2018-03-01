# new ideas

- There are so many things stays unknown and waiting for me to explore.

## Thinking in the DenseNet and ResNet
- I want to put up a combination of DenseNet and ResNet. Precisely, I want
128--------.--------.--------.-conv-+-128
    |      | |      | |      |      |
    --conv-- --conv-- --conv--      |
    |                               |
    ---------------------------------
- Why? Beacuse the way DenseNet organises data is something like this, but it's somehow more efficient than this.
I think the feature reuse is a valuable idea, but the concatenate way makes it much harder to handle #channels.
In the above graph, I think this model may be computationally expensive, but it worths a try.
- Is this model computationlly effecient? I think this model as a raw idea of the combination of DenseNet and ResNet.
I don't know whether this model is the correct solution of the problem that the variance of the activation of the last layer of a block is bigger than every other layer in the same block.
- In fact this is just using the residual way to solve the #channel problem.
- There are many errors/mistakes in my implementation. Need to be corrected.