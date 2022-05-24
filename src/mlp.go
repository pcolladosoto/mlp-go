package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Mlp struct {
	InDim     int
	HiddenDim []int
	NHidden   int
	OutDim    int
	ActFunc   func(int, int, float64) float64
	Weights   []mat.Matrix
}

func NewMlp(dims []int, actF func(int, int, float64) float64, variance float64) (*Mlp, error) {
	if len(dims) < 3 {
		return nil, fmt.Errorf("we need at least 3 dimensions for the input, hidden and output layer")
	}

	mlp := Mlp{InDim: dims[0], HiddenDim: dims[1 : len(dims)-1], NHidden: len(dims) - 2, OutDim: dims[len(dims)-1], ActFunc: actF}

	// Let's avoid recomputing the standard deviation over and over
	stdDev := math.Sqrt(variance)
	for i := 0; i < mlp.NHidden+1; i++ {
		weights := make([]float64, dims[i+1]*(dims[i]+1))
		for j := range weights {
			weights[j] = rand.NormFloat64() * stdDev
		}
		mlp.Weights = append(mlp.Weights, mat.NewDense(dims[i+1], dims[i]+1, weights))
	}

	return &mlp, nil
}

func (mlp *Mlp) SetWeights(init_ws [][]float64) {
	for i, w := range init_ws {
		r, c := mlp.Weights[i].Dims()
		mlp.Weights[i] = mat.NewDense(r, c, w)
	}
}

func (mlp *Mlp) String() string {
	msg := fmt.Sprintf("MLP Description:\n\tDimensions       -> %v / %v / %v\n", mlp.InDim, mlp.HiddenDim, mlp.OutDim)
	for i, w := range mlp.Weights {
		msg += fmt.Sprintf("\tWeight Matrix %2d -> %v\n", i, mat.Formatted(w, mat.FormatMATLAB()))
	}
	return msg
}

func (mlp *Mlp) ComputeActivation(input []float64) (activations []*mat.Dense, net_activations []*mat.Dense) {
	var net_acts, acts []*mat.Dense

	acts = append(acts, mat.NewDense(mlp.InDim, 1, input))

	for i, w := range mlp.Weights {
		rawM := acts[i].RawMatrix()
		tmpA := mat.NewDense(rawM.Rows+1, rawM.Cols, append(rawM.Data, 1))

		// fmt.Printf("Multiplying %v * %v\n", mat.Formatted(w, mat.FormatMATLAB()), mat.Formatted(tmpA, mat.FormatMATLAB()))
		// fmt.Printf("Dimensionality:\n\tWeights -> %s\n\tActs    -> %s\n", fmt.Sprint(w.Dims()), fmt.Sprint(tmpA.Dims()))

		var tmp mat.Dense
		tmp.Mul(w, tmpA)

		net_acts = append(net_acts, mat.DenseCopyOf(&tmp))

		acts = append(acts, new(mat.Dense))

		acts[i+1].Apply(mlp.ActFunc, &tmp)
	}

	return acts[1:], net_acts
}

func (mlp *Mlp) GenTestData() {
	mlp.SetWeights([][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}})

	acts, net_acts := mlp.ComputeActivation([]float64{1, 0})

	acts_buff, net_acts_buff := bytes.Buffer{}, bytes.Buffer{}

	fmt.Printf("Activations:\n")
	for i, act := range acts {
		fmt.Printf("\tActivation     %2d -> %6.3f\n", i, mat.Formatted(act, mat.FormatMATLAB()))
		act.MarshalBinaryTo(&acts_buff)
	}

	fmt.Printf("\nNet Activations:\n")
	for i, net_act := range net_acts {
		fmt.Printf("\tNet Activation %2d -> %6.3f\n", i, mat.Formatted(net_act, mat.FormatMATLAB()))
		net_act.MarshalBinaryTo(&net_acts_buff)
	}

	ioutil.WriteFile("testdata/act_data.b64", []byte(base64.StdEncoding.EncodeToString(acts_buff.Bytes())), 0644)
	ioutil.WriteFile("testdata/net_act_data.b64", []byte(base64.StdEncoding.EncodeToString(net_acts_buff.Bytes())), 0644)
}
