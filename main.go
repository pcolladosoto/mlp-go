package main

import (
	"fmt"
	"math/rand"
)

func main() {
	// fmt.Printf("Loading labels...\n")
	// labels, err := read_labels("../data/train-labels-idx1-ubyte")
	// if err != nil {
	// 	fmt.Printf("Error reading the labels: %v\n", err)
	// 	os.Exit(-1)
	// }

	// fmt.Printf(
	// 	"\tMagic number: %d (%#x)\n\tData type: %s\n\tDimensionality: %d\n\tN labels: %d (%#x)\n\tFirst 5 labels: %v\n",
	// 	labels.magic_num, labels.magic_num, labels.data_type, labels.dimensionality, labels.n, labels.n, labels.labels[:10],
	// )

	// fmt.Printf("\nLoading images...\n")
	// imgs, err := read_imgs("../data/train-images-idx3-ubyte")
	// if err != nil {
	// 	fmt.Printf("Error reading the images: %v\n", err)
	// 	os.Exit(-1)
	// }

	// fmt.Printf(
	// 	"\tMagic number: %d (%#x)\n\tData type: %s\n\tDimensionality: %d\n\tN images: %d (%#x)\n\tRows / image: %d (%#x)\n\tCols / image: %d (%#x)\n\tFirst image: %v\n",
	// 	imgs.magic_num, imgs.magic_num, imgs.data_type, imgs.dimensionality, imgs.n, imgs.n, imgs.img_rows, imgs.img_rows, imgs.img_cols, imgs.img_cols, imgs.images[0],
	// )

	// fmt.Printf("\nStoring the 20 first images as PNG files...\n")
	// for i := 0; i < 20; i++ {
	// 	if err := dump_image(imgs, i, fmt.Sprintf("../data/imgs/img-%d.png", i)); err != nil {
	// 		fmt.Printf("Error generating the image: %v\n", err)
	// 	}
	// }

	dsize := 80.0

	train_passes := 1000000

	fmt.Printf("\nGenerating some XOR data...\n")
	xorData, xorLabels := genXor(int(dsize), 0.1)
	// for i, p := range xorData {
	// 	fmt.Printf("\tData point %2d (Out -> %1.0f) = %6.3f\n", i, xorOutput(p), mat.Formatted(p))
	// }

	xorDataTrain, xorLabelsTrain := xorData[:int(dsize*0.9)], xorLabels[:int(dsize*0.9)]
	xorDataTest, xorLabelsTest := xorData[int(dsize*0.9):], xorLabels[int(dsize*0.9):]

	var outputPredTest []float64

	m, _ := NewMlp([]int{2, 2, 1}, Sigmoid, 1)
	// m.SetWeights([][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}})
	fmt.Printf("%s", m)

	// acts, net_acts := m.ComputeActivation([]float64{1, 0})

	// fmt.Printf("\nActivations for [1; 0]:\n")
	// for i, act := range acts {
	// 	fmt.Printf("\tActivation     %2d -> %6.3f\n", i, mat.Formatted(act, mat.FormatMATLAB()))
	// }

	// fmt.Printf("\nNet Activations for [1; 0]:\n")
	// for i, net_act := range net_acts {
	// 	fmt.Printf("\tNet Activation %2d -> %6.3f\n", i, mat.Formatted(net_act, mat.FormatMATLAB()))
	// }

	for i := 0; i < train_passes; i++ {
		rSample := rand.Intn(int(dsize * 0.9))

		m.Adapt(xorDataTrain[rSample], []float64{xorLabelsTrain[rSample]}, 0.05)

		// output, _, _ := m.ComputeActivation(xorDataTrain[rSample].RawMatrix().Data)

		// if i < int(0.1*float64(train_passes)) || i > int(0.9*float64(train_passes)) {
		// 	fmt.Printf("Training output for %6.3f: %6.3f [%d]\n",
		// 		mat.Formatted(xorDataTrain[rSample], mat.FormatMATLAB()), output[0], int(xorLabelsTrain[rSample]))
		// }
	}

	for i, dp := range xorDataTest {
		output, _, _ := m.ComputeActivation(dp)

		if output[0] > 0.5 {
			outputPredTest = append(outputPredTest, 1)
		} else {
			outputPredTest = append(outputPredTest, 0)
		}

		fmt.Printf("Testing output for %6.3f: [%6.3f; %6.3f] [%d] [%d]\n",
			dp[0], dp[1], output[0], int(outputPredTest[i]), int(xorLabelsTest[i]))
	}

	errs := 0.0
	for i, p_res := range outputPredTest {
		if p_res != xorLabelsTest[i] {
			errs++
		}
	}

	fmt.Printf("Testing error rate: %2.5f\n", errs/float64(len(outputPredTest)))
}
