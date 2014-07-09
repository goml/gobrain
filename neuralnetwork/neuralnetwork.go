package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"
)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func matrix(I, J int) [][]float64 {
	m := make([][]float64, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float64, J)
	}
	return m
}

func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

type NeuralNetwork struct {
	// Number of input, hidden and output nodes
	NI, NH, NO int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	AI, AH, AO []float64
	// Weights
	WI, WO [][]float64
	// Last change in weights for momentum
	CI, CO [][]float64
}

func New(NI, NH, NO int, Regression bool) *NeuralNetwork {
	nn := &NeuralNetwork{NI: NI + 1, NH: NH + 1, NO: NO, Regression: Regression}

	nn.AI = vector(nn.NI, 1.0)
	nn.AH = vector(nn.NH, 1.0)
	nn.AO = vector(nn.NO, 1.0)

	nn.WI = matrix(nn.NI, nn.NH)
	nn.WO = matrix(nn.NH, nn.NO)

	for i := 0; i < nn.NI; i++ {
		for j := 0; j < nn.NH; j++ {
			nn.WI[i][j] = random(-1, 1)
		}
	}

	for i := 0; i < nn.NH; i++ {
		for j := 0; j < nn.NO; j++ {
			nn.WO[i][j] = random(-1, 1)
		}
	}

	nn.CI = matrix(nn.NI, nn.NH)
	nn.CO = matrix(nn.NH, nn.NO)

	return nn
}

func (nn *NeuralNetwork) Update(inputs []float64) []float64 {
	if len(inputs) != nn.NI-1 {
		fmt.Println("Error: wrong number of inputs")
		return []float64{} // should return error
	}

	for i := 0; i < nn.NI-1; i++ {
		nn.AI[i] = inputs[i]
	}

	for i := 0; i < nn.NH-1; i++ {
		var sum float64 = 0.0
		for j := 0; j < nn.NI; j++ {
			sum += nn.AI[j] * nn.WI[j][i]
		}
		nn.AH[i] = sigmoid(sum)
	}

	for i := 0; i < nn.NO; i++ {
		var sum float64 = 0.0
		for j := 0; j < nn.NH; j++ {
			sum += nn.AH[j] * nn.WO[j][i]
		}
		if nn.Regression {
			nn.AO[i] = sum
		} else {
			nn.AO[i] = sigmoid(sum)
		}
	}

	return nn.AO
}

func (nn *NeuralNetwork) BackPropagate(targets []float64, lRate, mFactor float64) float64 {
	if len(targets) != nn.NO {
		fmt.Println("Error: wrong number of target values")
		return 0.0
	}

	output_deltas := vector(nn.NO, 0.0)
	for i := 0; i < nn.NO; i++ {
		output_deltas[i] = targets[i] - nn.AO[i]

		if !nn.Regression {
			output_deltas[i] = dsigmoid(nn.AO[i]) * output_deltas[i]
		}
	}

	hidden_deltas := vector(nn.NH, 0.0)
	for i := 0; i < nn.NH; i++ {
		var e float64 = 0.0

		for j := 0; j < nn.NO; j++ {
			e += output_deltas[j] * nn.WO[i][j]
		}
		hidden_deltas[i] = dsigmoid(nn.AH[i]) * e
	}

	for i := 0; i < nn.NH; i++ {
		for j := 0; j < nn.NO; j++ {
			change := output_deltas[j] * nn.AH[i]
			nn.WO[i][j] = nn.WO[i][j] + lRate*change + mFactor*nn.CO[i][j]
			nn.CO[i][j] = change
		}
	}

	for i := 0; i < nn.NI; i++ {
		for j := 0; j < nn.NH; j++ {
			change := hidden_deltas[j] * nn.AI[i]
			nn.WI[i][j] = nn.WI[i][j] + lRate*change + mFactor*nn.CI[i][j]
			nn.CI[i][j] = change
		}
	}

	var e float64 = 0.0

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i]-nn.AO[i], 2)
	}

	return e
}

func (nn *NeuralNetwork) Train(patterns [][][]float64, iterations int, lRate, mFactor float64) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64 = 0.0
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackPropagate(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *NeuralNetwork) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}
