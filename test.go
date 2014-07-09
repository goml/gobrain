package main

import (
	"fmt"
	"math/rand"

	"./neuralnetwork"
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

	nn := neuralnetwork.New(2, 2, 1, false)

	fmt.Println(nn)

	nn.Train(patterns, 1000, 0.6, 0.4)

	nn.Test(patterns)

}
