# MLP implemented in Go
This repository contains a Go module implementing a *Multi Layer Perceptron* (i.e. MLP).

The implementation itself resides on the `mlp/` directory and can be imported into Go projects with:

```go
import "github.com/pcolladosoto/mlp-go/mlp"
```

We have also added a [Cobra](https://pkg.go.dev/github.com/spf13/cobra)-based executable containing example experiments leveraging our MLP implementation. You can find that under the `experiments/` directory.

This implementation relies heavily on [Gonum](https://www.gonum.org) for everything matrix-related. The internals shouldn't be visible to the end user, but we wanted to make it clear we haven't implemented the entire 'liner-algebra' engine.

## Experiments
Even though the MLP can be used for a myriad of tasks, we have included a couple of common classification experiments to both showcase and test the abilities of our MLP.

### XOR
When developing the MLP we relied on a XOR experiment to assess whether we were on the right track or not. The experiment itself is accessible through the `xor` subcommand of the experiment binary we described above.

In our experience, it doesn't take too much 'number-crunching' to get really good error rates given the low dimensionality of the problem.

### MNIST
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains a ton of handwritten digits. The MNIST experiment tries to classify them.

The experiment is also available within the binary and can be accessed through the `mnist` subcommand.

Our MLP module includes a series of functions dealing with the MNIST dataset itself: it's not provided in a standard format whatsoever. What's more, we've added some functions generating PNG images for an arbitrary entry in the MNIST dataset. That let's us take a 'look' at the data itself to understand the complexity of the problem at hand.

#### Dataset format
As seen [here](http://yann.lecun.com/exdb/mnist/), the dataset is structured as:

    # Label file
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    0004     32 bit integer  60000            number of items 
    0008     unsigned byte   ??               label 
    0009     unsigned byte   ??               label 
    ........ 
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    # Data file
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    0016     unsigned byte   ??               pixel 
    0017     unsigned byte   ??               pixel 
    ........ 
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

This is all based on the *IDX* file format, which can be loosely defined as:

    magic number 
    size in dimension 0 
    size in dimension 1 
    size in dimension 2 
    ..... 
    size in dimension N 
    data

In our case, the bottom line is we're dealing with an input dimension of `28 * 28 = 784`. Given we're classifying digits, we should also consider using an output dimension of exactly `10`. Please note metadata such as the *Magic Number* are interpreted by our module so as to provide meaningful information.
