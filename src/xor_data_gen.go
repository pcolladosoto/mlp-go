package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func genXor(n int, stdDev float64) (data_points []*mat.Dense, labels []float64) {
	data := make([]*mat.Dense, n)
	var lbls []float64
	var m mat.Dense
	for i := 0; i < n; i++ {
		switch i % 4 {
		case 0:
			data[i] = mat.NewDense(1, 2, []float64{0, 0})
		case 1:
			data[i] = mat.NewDense(1, 2, []float64{0, 1})
		case 2:
			data[i] = mat.NewDense(1, 2, []float64{1, 0})
		case 3:
			data[i] = mat.NewDense(1, 2, []float64{1, 1})
		}

		// Generate some normally distributed noise
		m.Scale(stdDev, mat.NewDense(1, 2, []float64{rand.NormFloat64(), rand.NormFloat64()}))
		data[i].Add(data[i], &m)
		lbls = append(lbls, xorOutput(data[i]))
	}
	return data, lbls
}

func xorOutput(in *mat.Dense) float64 {
	_, c := in.Dims()
	xor := 0
	for i := 0; i < c; i++ {
		if in.At(0, i) > 0.5 {
			xor++
		}
	}

	if xor == 1 {
		return 1
	}
	return 0
}
