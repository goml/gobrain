# gobrain

Neural Networks written in go

[![GoDoc](https://godoc.org/github.com/goml/gobrain?status.svg)](https://godoc.org/github.com/goml/gobrain)
[![Build Status](https://travis-ci.org/goml/gobrain.svg?branch=master)](https://travis-ci.org/goml/gobrain)

## Getting Started
The version `1.0.0` includes just basic Neural Network functions such as Feed Forward and Elman Recurrent Neural Network.
A simple Feed Forward Neural Network can be constructed and trained as follows:

```go
package main

import (
	"github.com/goml/gobrain"
	"math/rand"
)

func main() {
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
}

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

In the example folder there are runnable examples with persistence of the trained network on file.

In example/02 the network is saved on file and in example/03 the network is loaded from file.

To run the example cd in the folder and run

	go run main.go

## Recurrent Neural Network

This library implements Elman's Simple Recurrent Network.

To take advantage of this, one can use the `SetContexts` function.

```go
ff.SetContexts(1, nil)
```

In the example above, a single context will be created initilized with `0.5`. It is also possible
to create custom initilized contexts, for instance:

```go
contexts := [][]float64{
	{0.5, 0.8, 0.1}
}
```

Note that custom contexts must have the same size of hidden nodes + 1 (bias node),
in the example above the size of hidden nodes is 2, thus the context has 3 values.

## Changelog
* 1.0.0 - Added Feed Forward Neural Network with contexts from Elman RNN

