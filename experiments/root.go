package main

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/pcolladosoto/mlp-go/mlp"
)

var (
	mlpDims        []int
	actFunction    string
	weightVariance float64
	learningRate   float64

	actFuncMap map[string]func(float64) float64 = map[string]func(float64) float64{
		"sigmoid":  mlp.Sigmoid,
		"relu":     mlp.ReLu,
		"unitstep": mlp.UnitStep,
	}

	rootCmd = &cobra.Command{
		Use:   "mlp-experiment",
		Short: "A binary implementing classification experiments leveraging a MLP.",
		Long: "This executable implements some sample experiments driving the MLP implemented on github.com/pcolladosoto/mlp-go.\n" +
			"Each available experiment is provided through a sub-command.\n",
		Args: func(cmd *cobra.Command, args []string) error {
			if _, ok := actFuncMap[strings.ToLower(actFunction)]; !ok {
				return fmt.Errorf("wrong activation function %s: choose one of [sigmoid, ReLu, unitStep]", actFunction)
			}
			return nil
		},
	}
)

func init() {
	// Disable Cobra completions
	rootCmd.CompletionOptions.DisableDefaultCmd = true

	rootCmd.AddCommand(xorExp)

	rootCmd.PersistentFlags().IntSliceVar(&mlpDims, "mlp_dimensions", []int{2, 2, 1},
		"The dimension of each layer of the MLP. Note the initial and final dimensions are those of the input and output, respectively.")
	rootCmd.PersistentFlags().StringVar(&actFunction, "act_function", "sigmoid",
		"The activation function for each MLP neuron. One of: [sigmoid, ReLu, unitStep].")
	rootCmd.PersistentFlags().Float64Var(&weightVariance, "weight_variance", 1,
		"The variance for the random and normally-distributed initial weights.")
	rootCmd.PersistentFlags().Float64Var(&learningRate, "learning_rate", 0.05,
		"The learning rate for the back-propagation algorithm.")
}
