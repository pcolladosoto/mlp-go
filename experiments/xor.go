package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/pcolladosoto/mlp-go/mlp"
)

func init() {
	xorExp.Flags().IntVar(&dataSize, "data_size", 80, "The amount of data points to generate for the experiment.")
	xorExp.Flags().IntVar(&trainDataPercentage, "train_percentage", 90,
		"Percentage of the total data to use for training in the [0, 100) interval. The rest is used for testing.")
	xorExp.Flags().StringVar(&trainingMode, "training_mode", "online",
		"How to treat data points used for training. Once of: [online, batch].")

	xorExp.Flags().Float64Var(&xorStdDev, "std_deviation", 0.1,
		"The standard deviation of the noise added to generated XOR data.")
}

var (
	dataSize            int
	trainDataPercentage int
	trainingMode        string
	trainingPasses      int

	xorStdDev float64

	xorExp = &cobra.Command{
		Use:   "xor <training passes>",
		Short: "Use a MLP to classify 2-dimensional XOR data points.",
		Long: "This experiment generates XOR data and then trains the MLP on it.\n" +
			"You MUST provide the number of training iterations as an argument. The rest'\n" +
			"of the parameters are configured through flags. Feel free to use `-h` to take a look!\n",
		Args: func(cmd *cobra.Command, args []string) error {
			if dataSize < 0 {
				return fmt.Errorf("you should provide a positive amount of data to generate")
			}
			if trainDataPercentage < 0 || trainDataPercentage > 100 {
				return fmt.Errorf("the training data percentage should be within the [0, 100) interval")
			}
			if trainingMode != "online" && trainingMode != "batch" {
				return fmt.Errorf("unsupported training mode %s. Choose either online or batch", trainingMode)
			}

			if len(args) != 1 {
				return fmt.Errorf("you just need to provide the number of training passes on the data")
			}
			tPasses, err := strconv.Atoi(args[0])
			if err != nil {
				return fmt.Errorf("couldn't parse the number of training passes: %v", err)
			}
			trainingPasses = tPasses

			return nil
		},
		Run: func(cmd *cobra.Command, args []string) {
			m, err := mlp.NewMlp(mlpDims, actFuncMap[strings.ToLower(actFunction)], weightVariance)
			if err != nil {
				fmt.Printf("couldn't instantiate an MLP: %v\n", err)
				os.Exit(-1)
			}
			fmt.Printf("%s", m)

			fmt.Printf("\nGenerating XOR data... ")
			xorData, xorLabels := mlp.GenXor(dataSize, xorStdDev)

			trainDataThreshold := int(float64(dataSize) * (float64(trainDataPercentage) / 100.0))

			xorDataTrain, xorLabelsTrain := xorData[:trainDataThreshold], xorLabels[:trainDataThreshold]
			xorDataTest, xorLabelsTest := xorData[trainDataThreshold:], xorLabels[trainDataThreshold:]

			fmt.Printf("done!\n")

			var outputPredTest []float64

			fmt.Printf("\nTraining the MLP... ")
			for i := 0; i < trainingPasses; i++ {
				rSample := rand.Intn(trainDataThreshold)
				m.Adapt(xorDataTrain[rSample], []float64{xorLabelsTrain[rSample]}, learningRate)
			}
			fmt.Printf("done!\n")

			tr := "+ ------------------------------------------- +"

			fmt.Printf("\nTESTING RESULTS:\n\t%s\n", tr)
			for i, dp := range xorDataTest {
				output, _, _ := m.ComputeActivation(dp)

				if output[0] > 0.5 {
					outputPredTest = append(outputPredTest, 1)
				} else {
					outputPredTest = append(outputPredTest, 0)
				}

				fmt.Printf("\t| Output for [%6.3f; %6.3f]: %6.3f [%d] [%d] |\n",
					dp[0], dp[1], output[0], int(outputPredTest[i]), int(xorLabelsTest[i]))
			}

			fmt.Printf("\t%s\n\t|       TESTING ERROR RATE -> %2.5f         |\n\t%s\n",
				tr, mlp.ErrorRate(outputPredTest, xorLabelsTest), tr)
		},
	}
)
