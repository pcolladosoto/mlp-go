package main

import (
	"math/rand"
)

func genXor(n int, stdDev float64) (data_points [][]float64, labels []float64) {
	data := make([][]float64, n)
	var lbls []float64
	for i := 0; i < n; i++ {
		switch i % 4 {
		case 0:
			data[i] = []float64{0, 0}
		case 1:
			data[i] = []float64{0, 1}
		case 2:
			data[i] = []float64{1, 0}
		case 3:
			data[i] = []float64{1, 1}
		}

		// Generate some normally distributed noise
		data[i][0] += stdDev * rand.NormFloat64()
		data[i][1] += stdDev * rand.NormFloat64()

		lbls = append(lbls, xorOutput(data[i]))
	}
	return data, lbls
}

func xorOutput(in []float64) float64 {
	xor := 0
	for i := 0; i < 2; i++ {
		if in[i] > 0.5 {
			xor++
		}
	}
	if xor == 1 {
		return 1
	}
	return 0
}
