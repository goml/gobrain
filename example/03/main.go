package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/goml/gobrain"
	"github.com/goml/gobrain/persist"
)

func main() {
	// set the random seed to 0
	rand.Seed(0)

	filename := "../02/ff.network"

	// instantiate the Feed Forward
	ff := &gobrain.FeedForward{}

	err := persist.Load(filename, &ff)
	if err != nil {
		log.Println("impossible to load network from file: ", err.Error())
	}

	// sends inputs to the neural network
	inputs := []float64{1, 1}

	// saves the result
	result := ff.Update(inputs)

	// prints the result
	fmt.Println(result)

}
