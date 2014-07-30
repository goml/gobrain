#gobrain

Neural Networks written in go


## Getting Started
The version `1.0.0` includes just basic Neural Network functions such as Feed Forward and Elman Recurrent Neural Network.
A simple Feed Forward Neural Network can be constructed and trained as follows:

```go
// set the random seed to 0
rand.Seed(0)

// create the XOR representation patter to train the network
patterns := [][][]float64{
  {{0, 0}, {0}},
  {{0, 1}, {1}},
  {{1, 0}, {1}},
  {{1, 1}, {0}},
}

// instantiate the Feed Forward
ff := &gobrain.FeedForward{}

// initialize the Neural Network;
// the networks structure will contain:
// 2 inputs, 2 hidden nodes and 1 output.
ff.Init(2, 2, 1)

// train the network using the XOR patterns
// the training will run for 1000 epochs
// the learning rate is set to 0.6 and the momentum factor to 0.4
// use true in the last parameter to receive reports about the learning error
ff.Train(patterns, 1000, 0.6, 0.4, true)
```

After running this code the network will be trained and ready to be used.

The network can be tested running using the `Test` method, for instance:

```go
ff.Test(patterns)
```

The test operation will print in the console something like:

```
[0 0] -> [0.057503945708445]  :  [0]
[0 1] -> [0.930100635071210]  :  [1]
[1 0] -> [0.927809966227284]  :  [1]
[1 1] -> [0.097408795324620]  :  [0]
```

Where the first values are the inputs, the values after the arrow `->` are the output values from the network and the values after `:` are the expected outputs.

The method `Update` can be used to predict the output given an input, for example:

```go
inputs := []float64{1, 1}
ff.Update(inputs)
```

the output will be a vector with values ranging from `0` to `1`.

## Recurrent Neural Network


## Changelog
* 1.0.0 - Added Feed Forward Neural Network with contexts from Elman RNN

