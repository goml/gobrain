package gobrain

import (
    // "testing"
    "math/rand"
)

func ExampleSimpleFeedForward() {
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
    ff := &FeedForward{}

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 2 hidden nodes and 1 output.
    ff.Init(2, 2, 1)

    // train the network using the XOR patterns
    // the training will run for 1000 epochs
    // the learning rate is set to 0.6 and the momentum factor to 0.4
    // use true in the last parameter to receive reports about the learning error
    ff.Train(patterns, 1000, 0.6, 0.4, false)

    // testing the network
    ff.Test(patterns)

    // predicting a value
    inputs := []float64{1, 1}
    ff.Update(inputs)
    
    // Output:
    // [0 0] -> [0.057503945708445206]  :  [0]
    // [0 1] -> [0.9301006350712101]  :  [1]
    // [1 0] -> [0.9278099662272838]  :  [1]
    // [1 1] -> [0.09740879532462123]  :  [0]
}
