package main

import (
	// "fmt"
	"math/rand"

	"./nn"
)

func main() {
	//rand.Seed(time.Now().UTC().UnixNano())
	rand.Seed(0)
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	ff := &nn.FeedForward{}

	ff.Init(2, 2, 1)
	ff.Train(patterns, 1000, 0.6, 0.4, false)
	ff.Test(patterns)

	// j 0 0 1
	// o 0 1 0
	// n 0 1 1
	// a 1 0 0
	// s 1 0 1

	// 1 1 2 3 5 8
	// patternsrnn := [][][]float64{
	// 	{{0, 0, 1}, {0, 1, 0}},
	// 	{{0, 1, 0}, {0, 1, 1}},
	// 	{{0, 1, 1}, {1, 0, 0}},
	// 	{{1, 0, 0}, {1, 0, 1}},
	// 	{{1, 0, 1}, {0, 0, 0}},
	// }

	// patternsrnntest := [][][]float64{
	// 	{{0, 1, 0}, {0, 1, 1}}, // o
	// 	{{0, 1, 1}, {1, 0, 0}}, // n
	// 	{{1, 0, 0}, {1, 0, 1}}, // a
	// 	{{1, 0, 1}, {0, 0, 0}}, // s
	// }

	// rnn := &nn.FeedForward{}

	// rnn.Init(3, 2, 3, false)
	// rnn.SetContexts(2, nil)
	// rnn.Train(patternsrnn, 100000, 0.3, 0.01, true)
	// rnn.Test(patternsrnntest)
}
