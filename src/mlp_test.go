package main

import (
	"bytes"
	"encoding/base64"
	"io/ioutil"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestWeightInit(t *testing.T) {
	m, err := NewMlp([]int{2, 2, 1}, Sigmoid, 1)
	if err != nil {
		t.Errorf("NewMlp() returned an error: %v", err)
	}

	init_weights := [][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}}

	m.SetWeights(init_weights)

	for i, w := range m.Weights {
		r, c := w.Dims()
		tmp := mat.NewDense(r, c, init_weights[i])
		if !mat.Equal(w, tmp) {
			t.Errorf("mismatch in weight matrix %d: %6.3f != %6.3f",
				i, mat.Formatted(w, mat.FormatMATLAB()), mat.Formatted(tmp, mat.FormatMATLAB()))
		}
	}
}

func TestForwardPropagation(t *testing.T) {
	m, err := NewMlp([]int{2, 2, 1}, Sigmoid, 1)
	if err != nil {
		t.Errorf("NewMlp() returned an error: %v", err)
	}

	m.SetWeights([][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}})

	var buff bytes.Buffer

	acts, net_acts := m.ComputeActivation([]float64{1, 0})

	ref_acts_raw, err := ioutil.ReadFile("testdata/act_data.b64")
	if err != nil {
		t.Fatalf("error opening reference data for activations: %v", err)
	}

	ref_acts, err := base64.StdEncoding.DecodeString(string(ref_acts_raw))
	if err != nil {
		t.Fatalf("error decoding reference data for activations: %v", err)
	}

	buff.Write(ref_acts)

	for _, act := range acts {
		var tmp mat.Dense
		if _, err := tmp.UnmarshalBinaryFrom(&buff); err != nil {
			t.Fatalf("error unmarshalling an activation matrix: %v", err)
		}

		if !mat.Equal(act, &tmp) {
			t.Errorf("activation matrix mismatch: %6.3f != %6.3f",
				mat.Formatted(act, mat.FormatMATLAB()), mat.Formatted(&tmp, mat.FormatMATLAB()))
		}
	}

	ref_net_acts_raw, err := ioutil.ReadFile("testdata/net_act_data.b64")
	if err != nil {
		t.Fatalf("error opening reference data for net activations: %v", err)
	}

	ref_net_acts, err := base64.StdEncoding.DecodeString(string(ref_net_acts_raw))
	if err != nil {
		t.Fatalf("error decoding reference data for net activations: %v", err)
	}

	buff.Reset()
	buff.Write(ref_net_acts)

	for _, net_act := range net_acts {
		var tmp mat.Dense
		if _, err := tmp.UnmarshalBinaryFrom(&buff); err != nil {
			t.Fatalf("error unmarshalling a net activation matrix: %v", err)
		}

		if !mat.Equal(net_act, &tmp) {
			t.Errorf("net activation matrix mismatch: %6.3f != %6.3f",
				mat.Formatted(net_act, mat.FormatMATLAB()), mat.Formatted(&tmp, mat.FormatMATLAB()))
		}
	}
}
