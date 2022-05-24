package main

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func ReLu(x float64) float64 {
	return math.Max(0, x)
}

func UnitStep(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
