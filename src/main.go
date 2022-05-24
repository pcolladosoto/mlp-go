package main

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Printf("Loading labels...\n")
	labels, err := read_labels("../data/train-labels-idx1-ubyte")
	if err != nil {
		fmt.Printf("Error reading the labels: %v\n", err)
		os.Exit(-1)
	}

	fmt.Printf(
		"\tMagic number: %d (%#x)\n\tData type: %s\n\tDimensionality: %d\n\tN labels: %d (%#x)\n\tFirst 5 labels: %v\n",
		labels.magic_num, labels.magic_num, labels.data_type, labels.dimensionality, labels.n, labels.n, labels.labels[:10],
	)

	fmt.Printf("\nLoading images...\n")
	imgs, err := read_imgs("../data/train-images-idx3-ubyte")
	if err != nil {
		fmt.Printf("Error reading the images: %v\n", err)
		os.Exit(-1)
	}

	fmt.Printf(
		"\tMagic number: %d (%#x)\n\tData type: %s\n\tDimensionality: %d\n\tN images: %d (%#x)\n\tRows / image: %d (%#x)\n\tCols / image: %d (%#x)\n\tFirst image: %v\n",
		imgs.magic_num, imgs.magic_num, imgs.data_type, imgs.dimensionality, imgs.n, imgs.n, imgs.img_rows, imgs.img_rows, imgs.img_cols, imgs.img_cols, imgs.images[0],
	)

	fmt.Printf("\nStoring the 20 first images as PNG files...\n")
	for i := 0; i < 20; i++ {
		if err := dump_image(imgs, i, fmt.Sprintf("../data/imgs/img-%d.png", i)); err != nil {
			fmt.Printf("Error generating the image: %v\n", err)
		}
	}

	fmt.Printf("\nGenerating some XOR data...\n")
	xorData := genXor(16, 1)
	for i, p := range xorData {
		fmt.Printf("\tData point %2d (Out -> %1.0f) = %6.3f\n", i, xorOutput(p), mat.Formatted(p))
	}

	m, _ := NewMlp([]int{2, 2, 1}, Sigmoid, 1)

	m.SetWeights([][]float64{{6, 0, -2, 2, -2, 0}, {-4, 2, 2}})

	fmt.Printf("\n%s", m)
}
