package tensorflow.layers

import tensorflow.D4PlusD1Array
import tensorflow.activations.Activation
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.KernelSize
import kotlin.math.sqrt
import kotlin.math.ceil
import kotlin.random.Random

class Conv2D(
    val inputShape: IntArray, // Input shape: [height, width, channels]
    private val filters: Int, // Number of filters (output channels)
    private val kernelSize: KernelSize, // Kernel size: [height, width]
    private val strides: IntArray = intArrayOf(1, 1), // Strides: [height, width]
    private val padding: String = "valid", // Padding: "valid" or "same"
    private val dilationRate: IntArray = intArrayOf(1, 1), // Dilation rate: [height, width]todo
    private val random: Random,
    private val activation: Activation? = null
) : TrainableLayer {

    private val inputChannels: Int = inputShape[3] // Number of input channels

    // Weights and biases for the convolutional layer
    var weights: D4Array<Double> = mk.zeros<Double>(kernelSize.height, kernelSize.width, inputChannels, filters)
    var biases: D1Array<Double> = mk.zeros<Double>(filters)

    // Gradients for weights and biases
    var dweights: D4Array<Double> = mk.zeros<Double>(kernelSize.height, kernelSize.width, inputChannels, filters)
    var dbiases: D1Array<Double> = mk.zeros<Double>(filters)

    // Cache for inputs during forward pass (used in backward pass)
    private lateinit var inputs: NDArray<Double, *>

    private lateinit var convOutput: D4Array<Double>

    val outputShape: IntArray
        get() {
            val (inputHeight, inputWidth, _) = inputShape
            val (kernelHeight, kernelWidth) = kernelSize
            val (strideHeight, strideWidth) = strides
            val dilationHeight = dilationRate[0]
            val dilationWidth = dilationRate[1]

            // Calculate effective kernel size considering dilation
            val effectiveKernelHeight = kernelHeight + (kernelHeight - 1) * (dilationHeight - 1)
            val effectiveKernelWidth = kernelWidth + (kernelWidth - 1) * (dilationWidth - 1)

            // Calculate output height and width based on padding
            val outputHeight = when (padding.lowercase()) {
                "same" -> ceil(inputHeight.toDouble() / strideHeight).toInt()
                "valid" -> ceil((inputHeight - effectiveKernelHeight + 1).toDouble() / strideHeight).toInt()
                else -> throw IllegalArgumentException("Padding must be 'valid' or 'same'")
            }

            val outputWidth = when (padding.lowercase()) {
                "same" -> ceil(inputWidth.toDouble() / strideWidth).toInt()
                "valid" -> ceil((inputWidth - effectiveKernelWidth + 1).toDouble() / strideWidth).toInt()
                else -> throw IllegalArgumentException("Padding must be 'valid' or 'same'")
            }

            return intArrayOf(outputHeight, outputWidth, filters)
        }

    init {
        // He initialization for weights
        val scale = sqrt(2.0 / (inputChannels * kernelSize.height * kernelSize.width))
        weights = mk.ndarray(
            List(kernelSize.height) {
                List(kernelSize.width) {
                    List(inputChannels) {
                        List(filters) { random.nextDouble(-scale, scale) }
                    }
                }
            }
        )

        // Initialize biases to a small positive value
        biases = mk.zeros<Double>(filters).map { 0.01 }
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(inputs.shape.size == 4) { "inputs shape must be [batch, height, width, channels] but passed ${inputs.shape.contentToString()}" }
        // Cache the inputs for use in the backward pass
        this.inputs = inputs

        // Perform the convolution operation with appropriate padding
        convOutput = if (padding == "same") {
            val paddedInput = padInput(inputs as D4Array<Double>, kernelSize, strides)
            convolve(paddedInput, weights, strides)
        } else {
            convolve(inputs as D4Array<Double>, weights, strides)
        }.D4PlusD1Array(biases)
        // Apply activation function if provided
        return activation?.forward(convOutput) ?: convOutput
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        require(dvalues.shape.size == 4) { "dvalues shape must be [batch, height, width, channels] but passed ${dvalues.shape.contentToString()}" }

        // Gradients for activation function (if any)
        val dactivation = if (activation != null) {
            // Use the output from the forward pass instead of recalculating
            activation.backward(dvalues, convOutput) // Use convOutput from forward
        } else {
            dvalues
        }

        // Gradients for weights
        dweights = convolveBackwardWeights(inputs as D4Array<Double>, dactivation as D4Array<Double>, strides)

        // Gradients for biases (sum over the batch and spatial axes)
        dbiases = mk.math.sum(
            mk.math.sum(
                mk.math.sum(dactivation, axis = 0),
                axis = 0
            ),
            axis = 0
        )

        // Gradients for inputs
        val dinputs = convolveBackwardInputs(dactivation, weights, strides)

        return dinputs
    }

    private fun padInput(input: D4Array<Double>, kernelSize: KernelSize, strides: IntArray): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = input.shape
        val (strideHeight, strideWidth) = strides
        val (kernelHeight, kernelWidth) = kernelSize

        // Calculate output dimensions for "same" padding
        val outputHeight = ceil(inputHeight.toDouble() / strideHeight).toInt()
        val outputWidth = ceil(inputWidth.toDouble() / strideWidth).toInt()

        // Calculate total padding required
        val totalPadHeight = ((outputHeight - 1) * strideHeight + kernelHeight - inputHeight).coerceAtLeast(0)
        val totalPadWidth = ((outputWidth - 1) * strideWidth + kernelWidth - inputWidth).coerceAtLeast(0)

        // Split total padding into start and end (asymmetric if needed)
        val padTop = totalPadHeight / 2
        val padBottom = totalPadHeight - padTop
        val padLeft = totalPadWidth / 2
        val padRight = totalPadWidth - padLeft

        // Apply asymmetric padding
        return pad(input, padTop, padBottom, padLeft, padRight)
    }

    private fun pad(
        input: D4Array<Double>,
        padTop: Int,
        padBottom: Int,
        padLeft: Int,
        padRight: Int
    ): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = input.shape
        val paddedHeight = inputHeight + padTop + padBottom
        val paddedWidth = inputWidth + padLeft + padRight

        // Initialize a new tensor with the padded dimensions
        val paddedInput = mk.zeros<Double>(batchSize, paddedHeight, paddedWidth, inputChannels)

        // Copy the original input tensor into the padded tensor
        for (b in 0 until batchSize) {
            for (h in 0 until inputHeight) {
                for (w in 0 until inputWidth) {
                    for (c in 0 until inputChannels) {
                        paddedInput[b, h + padTop, w + padLeft, c] = input[b, h, w, c]
                    }
                }
            }
        }

        return paddedInput
    }

    private fun convolve(
        inputs: D4Array<Double>, // Input tensor: [batch, height, width, channels]
        weights: D4Array<Double>, // Filters: [kernelHeight, kernelWidth, inputChannels, filters]
        strides: IntArray // Strides: [strideHeight, strideWidth]
    ): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = inputs.shape
        val (kernelHeight, kernelWidth, _, filters) = weights.shape
        val (strideHeight, strideWidth) = strides

        // Calculate output dimensions (no padding logic here)
        val outputHeight = (inputHeight - kernelHeight) / strideHeight + 1
        val outputWidth = (inputWidth - kernelWidth) / strideWidth + 1

        // Initialize output tensor
        val output = mk.zeros<Double>(batchSize, outputHeight, outputWidth, filters)

        // Perform convolution
        for (b in 0 until batchSize) {
            for (oh in 0 until outputHeight) {
                for (ow in 0 until outputWidth) {
                    for (oc in 0 until filters) {
                        var sum = 0.0
                        for (kh in 0 until kernelHeight) {
                            for (kw in 0 until kernelWidth) {
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (ic in 0 until inputChannels) {
                                        // Use TensorFlow's weight shape: [kernelHeight, kernelWidth, inputChannels, filters]
                                        sum += inputs[b, ih, iw, ic] * weights[kh, kw, ic, oc]
                                    }
                                }
                            }
                        }
                        output[b, oh, ow, oc] = sum
                    }
                }
            }
        }

        return output
    }

    private fun convolveBackwardWeights(
        inputs: D4Array<Double>,
        dvalues: D4Array<Double>,
        strides: IntArray
    ): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = inputs.shape
        val (_, outputHeight, outputWidth, filters) = dvalues.shape
        val (strideHeight, strideWidth) = strides

        // Initialize gradients for weights with TensorFlow's shape
        val dweights = mk.zeros<Double>(kernelSize.height, kernelSize.width, inputChannels, filters)

        // Perform convolution (adjust loop indices to match TensorFlow's format)
        for (kh in 0 until kernelSize.height) {
            for (kw in 0 until kernelSize.width) {
                for (ic in 0 until inputChannels) {
                    for (oc in 0 until filters) {
                        for (oh in 0 until outputHeight) {
                            for (ow in 0 until outputWidth) {
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (b in 0 until batchSize) {
                                        dweights[kh, kw, ic, oc] +=
                                            inputs[b, ih, iw, ic] * dvalues[b, oh, ow, oc]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return dweights
    }

    fun convolveBackwardInputs(
        dvalues: D4Array<Double>, // Gradient of loss w.r.t. output: [batch, outputHeight, outputWidth, filters]
        weights: D4Array<Double>, // Filters: [kernelHeight, kernelWidth, inputChannels, filters]
        strides: IntArray // Strides: [strideHeight, strideWidth]
    ): D4Array<Double> {
        val (batchSize, outputHeight, outputWidth, _) = dvalues.shape
        val (strideHeight, strideWidth) = strides
        val (kernelHeight, kernelWidth, inputChannels, filters) = weights.shape

        // Use original input dimensions cached during forward pass
        val inputHeight = inputs.shape[1]
        val inputWidth = inputs.shape[2]

        // Initialize gradients for inputs with the correct channels
        val dinputs = mk.zeros<Double>(batchSize, inputHeight, inputWidth, inputChannels)

        // Perform gradient accumulation
        for (b in 0 until batchSize) {
            for (oh in 0 until outputHeight) {
                for (ow in 0 until outputWidth) {
                    for (oc in 0 until filters) {
                        for (kh in 0 until kernelHeight) {
                            for (kw in 0 until kernelWidth) {
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (ic in 0 until inputChannels) {
                                        // Use TensorFlow's weight shape: [kernelHeight, kernelWidth, inputChannels, filters]
                                        dinputs[b, ih, iw, ic] += dvalues[b, oh, ow, oc] * weights[kh, kw, ic, oc]
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
}