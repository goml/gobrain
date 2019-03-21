package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/axamon/gobrain"
	"github.com/axamon/gobrain/persist"
)

func main() {
	// set the random seed to 0
	rand.Seed(0)

	filaneme := "../02/ff.network"

	// instantiate the Feed Forward
	ff := &gobrain.FeedForward{}

	err := persist.Load(filaneme, &ff)
	if err != nil {
		log.Println("impossible to save network on file: ", err.Error())
	}

	// sends inputs to the neular network
	inputs := []float64{1, 1}

	// saves the result
	result := ff.Update(inputs)

	// prints the result
	fmt.Println(result)

}
