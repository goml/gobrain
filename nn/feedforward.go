package nn

import (
	"fmt"
	"log"
	"math"
)

type FeedForward struct {
	// Number of input, hidden and output nodes
	NInputs, NHiddens, NOutputs int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	InputActivations, HiddenActivations, OutputActivations []float64
	// ElmanRNN contexts
	Contexts [][]float64
	// Weights
	InputWeights, OutputWeights [][]float64
	// Last change in weights for momentum
	InputChanges, OutputChanges [][]float64
}

// Initialize the neural network
func (nn *FeedForward) Init(inputs, hiddens, outputs int) {
	nn.NInputs = inputs + 1   // +1 for bias
	nn.NHiddens = hiddens + 1 // +1 for bias
	nn.NOutputs = outputs

	nn.InputActivations = vector(nn.NInputs, 1.0)
	nn.HiddenActivations = vector(nn.NHiddens, 1.0)
	nn.OutputActivations = vector(nn.NOutputs, 1.0)

	nn.InputWeights = matrix(nn.NInputs, nn.NHiddens)
	nn.OutputWeights = matrix(nn.NHiddens, nn.NOutputs)

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[i][j] = random(-1, 1)
		}
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[i][j] = random(-1, 1)
		}
	}

	nn.InputChanges = matrix(nn.NInputs, nn.NHiddens)
	nn.OutputChanges = matrix(nn.NHiddens, nn.NOutputs)
}

func (nn *FeedForward) SetContexts(nContexts int, initValues [][]float64) {
	if initValues == nil {
		initValues = make([][]float64, nContexts)

		for i := 0; i < nContexts; i++ {
			initValues[i] = vector(nn.NHiddens, 0.5)
		}
	}

	nn.Contexts = initValues
}

func (nn *FeedForward) Update(inputs []float64) []float64 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NInputs-1; i++ {
		nn.InputActivations[i] = inputs[i]
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		var sum float64 = 0.0

		for j := 0; j < nn.NInputs; j++ {
			sum += nn.InputActivations[j] * nn.InputWeights[j][i]
		}

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		nn.HiddenActivations[i] = sigmoid(sum)
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	for i := 0; i < nn.NOutputs; i++ {
		var sum float64 = 0.0
		for j := 0; j < nn.NHiddens; j++ {
			sum += nn.HiddenActivations[j] * nn.OutputWeights[j][i]
		}

		nn.OutputActivations[i] = sigmoid(sum)
	}

	return nn.OutputActivations
}

func (nn *FeedForward) BackPropagate(targets []float64, lRate, mFactor float64) float64 {
	if len(targets) != nn.NOutputs {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := vector(nn.NOutputs, 0.0)
	for i := 0; i < nn.NOutputs; i++ {
		outputDeltas[i] = dsigmoid(nn.OutputActivations[i]) * (targets[i] - nn.OutputActivations[i])
	}

	hiddenDeltas := vector(nn.NHiddens, 0.0)
	for i := 0; i < nn.NHiddens; i++ {
		var e float64 = 0.0

		for j := 0; j < nn.NOutputs; j++ {
			e += outputDeltas[j] * nn.OutputWeights[i][j]
		}

		hiddenDeltas[i] = dsigmoid(nn.HiddenActivations[i]) * e
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			change := outputDeltas[j] * nn.HiddenActivations[i]
			nn.OutputWeights[i][j] = nn.OutputWeights[i][j] + lRate*change + mFactor*nn.OutputChanges[i][j]
			nn.OutputChanges[i][j] = change
		}
	}

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			change := hiddenDeltas[j] * nn.InputActivations[i]
			nn.InputWeights[i][j] = nn.InputWeights[i][j] + lRate*change + mFactor*nn.InputChanges[i][j]
			nn.InputChanges[i][j] = change
		}
	}

	var e float64 = 0.0

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i]-nn.OutputActivations[i], 2)
	}

	return e
}

func (nn *FeedForward) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64 = 0.0
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackPropagate(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *FeedForward) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}
