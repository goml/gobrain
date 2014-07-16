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

	ff.Init(2, 2, 1, false)
	ff.Train(patterns, 1000, 0.6, 0.4, false)
	ff.Test(patterns)

}
