package main

import (
	"fmt"
	"math/rand"

	"github.com/jonhkr/gobrain/neuralnetwork"
)

func main() {
	//rand.Seed(time.Now().UTC().UnixNano())
	rand.Seed(0)
	patterns := [][][]float32{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	// patterns := [][][]float32{
	//  {{.0}, {.0 * .0}},
	//  {{.1}, {.1 * .1}},
	//  {{.2}, {.2 * .2}},
	//  {{.3}, {.3 * .3}},
	//  {{.4}, {.4 * .4}},
	// }

	nn := neuralnetwork.New(2, 4, 1, false)

	fmt.Println(nn)

	nn.Train(patterns, 1000, 0.5, 0.2)

	nn.Test(patterns)

}
