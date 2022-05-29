package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Mlp struct {
	InDim     int
	HiddenDim []int
	NHidden   int
	OutDim    int
	ActFunc   func(float64) float64
	Weights   []mat.Matrix
}

func NewMlp(dims []int, actF func(float64) float64, variance float64) (*Mlp, error) {
	if len(dims) < 3 {
		return nil, fmt.Errorf("we need at least 3 dimensions for the input, hidden and output layer")
	}

	mlp := Mlp{InDim: dims[0], HiddenDim: dims[1 : len(dims)-1], NHidden: len(dims) - 2, OutDim: dims[len(dims)-1], ActFunc: actF}

	rand.Seed(time.Now().Unix())

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

func (mlp *Mlp) appendRow(m *mat.Dense, n float64) *mat.Dense {
	rawM := m.RawMatrix()
	return mat.NewDense(m.RawMatrix().Rows+1, rawM.Cols, append(rawM.Data, n))
}

func (mlp *Mlp) chopRow(m *mat.Dense) *mat.Dense {
	rawM := m.RawMatrix()
	return mat.NewDense(m.RawMatrix().Rows-1, rawM.Cols, rawM.Data[:len(rawM.Data)-1])
}

func (mlp *Mlp) ComputeActivation(input []float64) (output []float64, activations []*mat.Dense, net_activations []*mat.Dense) {
	var net_acts, acts []*mat.Dense

	acts = append(acts, mat.NewDense(mlp.InDim, 1, input))

	for i, w := range mlp.Weights {
		tmpA := mlp.appendRow(acts[i], 1)

		// fmt.Printf("Multiplying %v * %v\n", mat.Formatted(w, mat.FormatMATLAB()), mat.Formatted(tmpA, mat.FormatMATLAB()))
		// fmt.Printf("Dimensionality:\n\tWeights -> %s\n\tActs    -> %s\n", fmt.Sprint(w.Dims()), fmt.Sprint(tmpA.Dims()))

		var tmp mat.Dense
		tmp.Mul(w, tmpA)

		net_acts = append(net_acts, mat.DenseCopyOf(&tmp))

		acts = append(acts, new(mat.Dense))

		acts[i+1].Apply(func(i, j int, v float64) float64 { return mlp.ActFunc(v) }, &tmp)
	}

	return acts[len(acts)-1].RawMatrix().Data, acts[1:], net_acts
}

func (mlp *Mlp) Adapt(input, target []float64, learning_rate float64) {
	_, acts, _ := mlp.ComputeActivation(input)

	// Reverse the activations
	for i, j := 0, len(acts)-1; i < j; i, j = i+1, j-1 {
		acts[i], acts[j] = acts[j], acts[i]
	}

	delta_helper := func(a, b *mat.Dense) *mat.Dense {
		var tmp mat.Dense
		tmp.Apply(func(i, j int, v float64) float64 { return 1 - v }, b)
		tmp.MulElem(&tmp, b)
		tmp.MulElem(&tmp, a)
		return &tmp
	}

	var (
		deltas []*mat.Dense
		tmp    mat.Dense
	)

	// fmt.Printf("\nBeginning adaptation with learning rate = %1.3f\n", learning_rate)

	acts = append(acts, mat.NewDense(mlp.InDim, 1, input))
	tmp.Sub(acts[0], mat.NewDense(mlp.OutDim, 1, target))
	deltas = append(deltas, delta_helper(&tmp, acts[0]))

	// fmt.Printf("\tdeltas[00] -> %7.4f\n", mat.Formatted(deltas[0], mat.FormatMATLAB()))

	for i := range mlp.Weights {
		var updated_weights, tmp_delta mat.Dense
		updated_weights.Mul(deltas[i], mlp.appendRow(acts[i+1], 1).T())
		updated_weights.Apply(func(i, j int, v float64) float64 { return learning_rate * v }, &updated_weights)

		// fmt.Printf("\t\tWeigth Matrix variation for layer %2d -> %6.3f\n",
		// 	len(mlp.Weights)-i, mat.Formatted(&updated_weights, mat.FormatMATLAB()))

		updated_weights.Sub(mlp.Weights[len(mlp.Weights)-(i+1)], &updated_weights)

		// tmp_delta.Mul(updated_weights.T(), deltas[i])
		tmp_delta.Mul(mlp.Weights[len(mlp.Weights)-(i+1)].T(), deltas[i])

		deltas = append(deltas, delta_helper(mlp.chopRow(&tmp_delta), acts[i+1]))

		// fmt.Printf("\tdeltas[%02d] -> %7.4f\n", i+1, mat.Formatted(deltas[i+1], mat.FormatMATLAB()))
		// fmt.Printf("\t\tWeigth Matrix %2d before adapting     -> %6.3f\n",
		// 	len(mlp.Weights)-(i+1), mat.Formatted(mlp.Weights[len(mlp.Weights)-(i+1)], mat.FormatMATLAB()))

		mlp.Weights[len(mlp.Weights)-(i+1)] = mat.DenseCopyOf(&updated_weights)

		// fmt.Printf("\t\tWeigth Matrix %2d after adapting      -> %6.3f\n",
		// 	len(mlp.Weights)-(i+1), mat.Formatted(mlp.Weights[len(mlp.Weights)-(i+1)], mat.FormatMATLAB()))
	}
}

func (mlp *Mlp) GenTestData() {
	mlp.SetWeights([][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}})

	_, acts, net_acts := mlp.ComputeActivation([]float64{1, 0})

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
