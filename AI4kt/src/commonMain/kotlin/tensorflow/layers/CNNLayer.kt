package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.D4PlusD1Array
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Activation
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.sqrt
import kotlin.random.Random

class CNNLayer(
    private val inputChannels: Int,
    private val outputChannels: Int,
    private val kernelSize: Int,
    private val stride: Int = 1,
    private val padding: Int = 0,
    private val random: Random,
    private val activation: Activation? = null
) : Layer {

    // Weights and biases for the convolutional layer
    var weights: D4Array<Double> = mk.zeros<Double>(outputChannels, inputChannels, kernelSize, kernelSize)
    var biases: D1Array<Double> = mk.zeros<Double>(outputChannels)

    // Gradients for weights and biases
    var dweights: D4Array<Double> = mk.zeros<Double>(outputChannels, inputChannels, kernelSize, kernelSize)
    var dbiases: D1Array<Double> = mk.zeros<Double>(outputChannels)

    // Cache for inputs during forward pass (used in backward pass)
    private lateinit var inputs: NDArray<Double, *>

    init {
        // He initialization for weights
        val scale = sqrt(2.0 / (inputChannels * kernelSize * kernelSize))
        weights = mk.ndarray(
            List(outputChannels) {
                List(inputChannels) {
                    List(kernelSize) {
                        List(kernelSize) { random.nextDouble(-scale, scale) }
                    }
                }
            }
        )

        // Initialize biases to a small positive value
        biases = mk.zeros<Double>(outputChannels).map { 0.01 }
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(inputs.shape.size == 4) { "inputs shape must be [M,N,B,V] but passed ${inputs.shape.contentToString()}" }
        // Cache the inputs for use in the backward pass
        this.inputs = inputs

        // Perform the convolution operation
        val output = convolve(
            inputs as D4Array<Double>, weights, stride, padding
        ).D4PlusD1Array(biases)

        // Apply activation function if provided
        return activation?.forward(output) ?: output
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        require(dvalues.shape.size == 4) { "dvalues shape must be [M,N,B,V] but passed ${dvalues.shape.contentToString()}" }
        // Gradients for activation function (if any)
        val dactivation = if (activation != null) {
            // Pass the output of the layer (before activation) to the activation's backward method
            val output = convolve(
                inputs as D4Array<Double>, weights, stride, padding
            ).D4PlusD1Array(biases)
            activation.backward(dvalues, output)
        } else {
            dvalues
        }

        // Gradients for weights
        dweights = convolveBackwardWeights(
            inputs as D4Array<Double>, dactivation as D4Array<Double>, stride, padding
        )

        // Gradients for biases (sum over the batch and spatial axes)
        dbiases = mk.math.sum(
            mk.math.sum(
                mk.math.sum(dactivation, axis = 0),
                axis = 1
            ),
            axis = 1
        )

        // Gradients for inputs
        val dinputs = convolveBackwardInputs(dactivation, weights, stride, padding)

        return dinputs
    }

    private fun convolve(input: D4Array<Double>, weights: D4Array<Double>, stride: Int, padding: Int): D4Array<Double> {
        val (batchSize, inputChannels, inputHeight, inputWidth) = input.shape
        val (_, _, kernelHeight, kernelWidth) = weights.shape

        // Calculate output dimensions
        val outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1
        val outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1

        // Validate input and weights shapes
        if (inputChannels != weights.shape[1] || kernelHeight != weights.shape[2] || kernelWidth != weights.shape[3]) {
            throw Exception(
                "Input and weights shapes must match following requirements: " +
                        "Input shape must have ${weights.shape[1]} input channels, " +
                        "Weights shape must have a kernel height of $kernelHeight and kernel width of $kernelWidth"
            )
        }

        // Initialize output array
        val output = mk.zeros<Double>(batchSize, outputChannels, outputHeight, outputWidth)

        // Pad the input if necessary
        val paddedInput = if (padding > 0) {
            padInput(input, padding)
        } else {
            input
        }

        // Perform the convolution
        for (b in 0 until batchSize) {
            for (oc in 0 until outputChannels) {
                for (oh in 0 until outputHeight) {
                    for (ow in 0 until outputWidth) {
                        var sum = 0.0
                        for (ic in 0 until inputChannels) {
                            for (kh in 0 until kernelHeight) {
                                for (kw in 0 until kernelWidth) {
                                    val ih = oh * stride + kh
                                    val iw = ow * stride + kw
                                    sum += paddedInput[b, ic, ih, iw] * weights[oc, ic, kh, kw]
                                }
                            }
                        }
                        output[b, oc, oh, ow] = sum
                    }
                }
            }
        }

        return output
    }

    private fun convolveBackwardWeights(
        input: D4Array<Double>,
        dvalues: D4Array<Double>,
        stride: Int,
        padding: Int
    ): D4Array<Double> {
        val (batchSize, inputChannels, inputHeight, inputWidth) = input.shape
        val (_, _, outputHeight, outputWidth) = dvalues.shape

        // Initialize gradients for weights
        val dweights = mk.zeros<Double>(outputChannels, inputChannels, kernelSize, kernelSize)

        // Pad the input if necessary
        val paddedInput = if (padding > 0) {
            padInput(input, padding)
        } else {
            input
        }

        // Compute gradients for weights
        for (oc in 0 until outputChannels) {
            for (ic in 0 until inputChannels) {
                // For each position in the kernel
                for (kh in 0 until kernelSize) {
                    for (kw in 0 until kernelSize) {
                        var sum = 0.0
                        // Loop over the entire output gradient
                        for (b in 0 until batchSize) {
                            for (oh in 0 until outputHeight) {
                                for (ow in 0 until outputWidth) {
                                    // Corresponding position in input
                                    val ih = oh * stride + kh - padding
                                    val iw = ow * stride + kw - padding
                                    if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                        sum += paddedInput[b, ic, ih, iw] * dvalues[b, oc, oh, ow]
                                    }
                                }
                            }
                        }
                        dweights[oc, ic, kh, kw] = sum // Store the accumulated sum
                    }
                }
            }
        }

        return dweights
    }

    private fun convolveBackwardInputs(
        dvalues: D4Array<Double>,
        weights: D4Array<Double>,
        stride: Int,
        padding: Int
    ): D4Array<Double> {
        val (batchSize, _, outputHeight, outputWidth) = dvalues.shape
        val (_, inputChannels, kernelHeight, kernelWidth) = weights.shape

        // Calculate input dimensions
        val inputHeight = (outputHeight - 1) * stride + kernelHeight - 2 * padding
        val inputWidth = (outputWidth - 1) * stride + kernelWidth - 2 * padding

        // Initialize gradients for inputs
        val dinputs = mk.zeros<Double>(batchSize, inputChannels, inputHeight, inputWidth)

        // Compute gradients for inputs
        for (b in 0 until batchSize) {
            for (ic in 0 until inputChannels) {
                for (oh in 0 until outputHeight) {
                    for (ow in 0 until outputWidth) {
                        for (kh in 0 until kernelHeight) {
                            for (kw in 0 until kernelWidth) {
                                val ih = oh * stride + kh - padding
                                val iw = ow * stride + kw - padding
                                if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                    for (oc in 0 until outputChannels) {
                                        dinputs[b, ic, ih, iw] += dvalues[b, oc, oh, ow] * weights[oc, ic, kh, kw]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return dinputs
    }

    private fun padInput(input: D4Array<Double>, padding: Int): D4Array<Double> {
        val (batchSize, inputChannels, inputHeight, inputWidth) = input.shape
        val paddedInput =
            mk.zeros<Double>(batchSize, inputChannels, inputHeight + 2 * padding, inputWidth + 2 * padding)

        for (b in 0 until batchSize) {
            for (ic in 0 until inputChannels) {
                for (ih in 0 until inputHeight) {
                    for (iw in 0 until inputWidth) {
                        paddedInput[b, ic, ih + padding, iw + padding] = input[b, ic, ih, iw]
                    }
                }
            }
        }

        return paddedInput
    }
}