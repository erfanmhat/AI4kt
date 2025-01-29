package tensorflow.layers

import tensorflow.D4PlusD1Array
import tensorflow.activations.Activation
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.sqrt
import kotlin.random.Random

class CNNLayer(
    inputShape: IntArray, // Input shape: [height, width, channels]
    private val filters: Int, // Number of filters (output channels)
    private val kernelSize: IntArray, // Kernel size: [height, width]
    private val strides: IntArray = intArrayOf(1, 1), // Strides: [height, width]
    private val padding: String = "valid", // Padding: "valid" or "same"
    private val random: Random,
    private val activation: Activation? = null
) : TrainableLayer {

    private val inputChannels: Int = inputShape[2] // Number of input channels
    private val outputChannels: Int = filters // Number of output channels

    // Weights and biases for the convolutional layer
    var weights: D4Array<Double> = mk.zeros<Double>(outputChannels, inputChannels, kernelSize[0], kernelSize[1])
    var biases: D1Array<Double> = mk.zeros<Double>(outputChannels)

    // Gradients for weights and biases
    var dweights: D4Array<Double> = mk.zeros<Double>(outputChannels, inputChannels, kernelSize[0], kernelSize[1])
    var dbiases: D1Array<Double> = mk.zeros<Double>(outputChannels)

    // Cache for inputs during forward pass (used in backward pass)
    private lateinit var inputs: NDArray<Double, *>

    init {
        // He initialization for weights
        val scale = sqrt(2.0 / (inputChannels * kernelSize[0] * kernelSize[1]))
        weights = mk.ndarray(
            List(outputChannels) {
                List(inputChannels) { // Ensure this matches the input channels
                    List(kernelSize[0]) {
                        List(kernelSize[1]) { random.nextDouble(-scale, scale) }
                    }
                }
            }
        )

        // Initialize biases to a small positive value
        biases = mk.zeros<Double>(outputChannels).map { 0.01 }
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(inputs.shape.size == 4) { "inputs shape must be [batch, height, width, channels] but passed ${inputs.shape.contentToString()}" }
        // Cache the inputs for use in the backward pass
        this.inputs = inputs

        // Perform the convolution operation
        val output = convolve(
            inputs as D4Array<Double>, weights, strides, padding
        ).D4PlusD1Array(biases)

        // Apply activation function if provided
        return activation?.forward(output) ?: output
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        require(dvalues.shape.size == 4) { "dvalues shape must be [batch, height, width, channels] but passed ${dvalues.shape.contentToString()}" }
        // Gradients for activation function (if any)
        val dactivation = if (activation != null) {
            // Pass the output of the layer (before activation) to the activation's backward method
            val output = convolve(
                inputs as D4Array<Double>, weights, strides, padding
            ).D4PlusD1Array(biases)
            activation.backward(dvalues, output)
        } else {
            dvalues
        }

        // Gradients for weights
        dweights = convolveBackwardWeights(
            inputs as D4Array<Double>, dactivation as D4Array<Double>, strides, padding
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
        val dinputs = convolveBackwardInputs(dactivation, weights, strides, padding)

        return dinputs
    }

    private fun convolve(
        inputs: D4Array<Double>, // Input tensor: [batch, height, width, channels]
        weights: D4Array<Double>, // Filters: [outputChannels, kernelHeight, kernelWidth, inputChannels]
        strides: IntArray, // Strides: [strideHeight, strideWidth]
        padding: String // Padding: "valid" or "same"
    ): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = inputs.shape
        val (outputChannels, kernelHeight, kernelWidth, _) = weights.shape
        val (strideHeight, strideWidth) = strides

        // Debugging: Print shapes
        println("Input shape: ${inputs.shape.contentToString()}")
        println("Weights shape: ${weights.shape.contentToString()}")

        // Check if input channels match
        if (inputChannels != weights.shape[3]) {
            throw IllegalArgumentException("Input channels (${inputChannels}) must match weights input channels (${weights.shape[3]})")
        }

        // Calculate output dimensions based on padding
        val (outputHeight, outputWidth) = when (padding.lowercase()) {
            "valid" -> {
                val outHeight = (inputHeight - kernelHeight) / strideHeight + 1
                val outWidth = (inputWidth - kernelWidth) / strideWidth + 1
                Pair(outHeight, outWidth)
            }

            "same" -> {
                val outHeight = (inputHeight + strideHeight - 1) / strideHeight
                val outWidth = (inputWidth + strideWidth - 1) / strideWidth
                Pair(outHeight, outWidth)
            }

            else -> throw IllegalArgumentException("Padding must be 'valid' or 'same'")
        }

        // Initialize output tensor
        val output = mk.zeros<Double>(batchSize, outputHeight, outputWidth, outputChannels)

        // Perform convolution
        for (b in 0 until batchSize) { // Iterate over batch
            for (oh in 0 until outputHeight) { // Iterate over output height
                for (ow in 0 until outputWidth) { // Iterate over output width
                    for (oc in 0 until outputChannels) { // Iterate over output channels (filters)
                        var sum = 0.0

                        // Iterate over kernel height and width
                        for (kh in 0 until kernelHeight) {
                            for (kw in 0 until kernelWidth) {
                                // Calculate input indices
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw

                                // Check if input indices are within bounds
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (ic in 0 until inputChannels) { // Iterate over input channels
                                        sum += inputs[b, ih, iw, ic] * weights[oc, kh, kw, ic]
                                    }
                                }
                            }
                        }

                        // Assign the result to the output tensor
                        output[b, oh, ow, oc] = sum
                    }
                }
            }
        }

        return output
    }

    private fun convolveBackwardWeights(
        inputs: D4Array<Double>, // Input tensor: [batch, height, width, channels]
        dvalues: D4Array<Double>, // Gradient of loss w.r.t. output: [batch, outputHeight, outputWidth, outputChannels]
        strides: IntArray, // Strides: [strideHeight, strideWidth]
        padding: String // Padding: "valid" or "same"
    ): D4Array<Double> {
        val (batchSize, inputHeight, inputWidth, inputChannels) = inputs.shape
        val (_, outputHeight, outputWidth, outputChannels) = dvalues.shape
        val (strideHeight, strideWidth) = strides

        // Initialize gradients for weights
        val dweights = mk.zeros<Double>(outputChannels, inputChannels, kernelSize[0], kernelSize[1])

        // Iterate over batch, output height, and output width
        for (b in 0 until batchSize) {
            for (oh in 0 until outputHeight) {
                for (ow in 0 until outputWidth) {
                    for (oc in 0 until outputChannels) {
                        // Iterate over kernel height and width
                        for (kh in 0 until kernelSize[0]) {
                            for (kw in 0 until kernelSize[1]) {
                                // Calculate input indices
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw

                                // Check if input indices are within bounds
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (ic in 0 until inputChannels) {
                                        // Accumulate gradients for weights
                                        dweights[oc, ic, kh, kw] += inputs[b, ih, iw, ic] * dvalues[b, oh, ow, oc]
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

    private fun convolveBackwardInputs(
        dvalues: D4Array<Double>, // Gradient of loss w.r.t. output: [batch, outputHeight, outputWidth, outputChannels]
        weights: D4Array<Double>, // Filters: [outputChannels, kernelHeight, kernelWidth, inputChannels]
        strides: IntArray, // Strides: [strideHeight, strideWidth]
        padding: String // Padding: "valid" or "same"
    ): D4Array<Double> {
        val (batchSize, outputHeight, outputWidth, outputChannels) = dvalues.shape
        val (_, kernelHeight, kernelWidth, inputChannels) = weights.shape
        val (strideHeight, strideWidth) = strides

        // Calculate input dimensions based on padding
        val (inputHeight, inputWidth) = when (padding.lowercase()) {
            "valid" -> {
                val inHeight = (outputHeight - 1) * strideHeight + kernelHeight
                val inWidth = (outputWidth - 1) * strideWidth + kernelWidth
                Pair(inHeight, inWidth)
            }

            "same" -> {
                val inHeight = outputHeight * strideHeight
                val inWidth = outputWidth * strideWidth
                Pair(inHeight, inWidth)
            }

            else -> throw IllegalArgumentException("Padding must be 'valid' or 'same'")
        }

        // Initialize gradients for inputs
        val dinputs = mk.zeros<Double>(batchSize, inputHeight, inputWidth, inputChannels)

        // Perform the backward pass for inputs
        for (b in 0 until batchSize) { // Iterate over batch
            for (oh in 0 until outputHeight) { // Iterate over output height
                for (ow in 0 until outputWidth) { // Iterate over output width
                    for (oc in 0 until outputChannels) { // Iterate over output channels (filters)
                        // Iterate over kernel height and width
                        for (kh in 0 until kernelHeight) {
                            for (kw in 0 until kernelWidth) {
                                // Calculate input indices
                                val ih = oh * strideHeight + kh
                                val iw = ow * strideWidth + kw

                                // Check if input indices are within bounds
                                if (ih < inputHeight && iw < inputWidth) {
                                    for (ic in 0 until inputChannels) { // Iterate over input channels
                                        // Accumulate gradients for inputs
                                        dinputs[b, ih, iw, ic] += dvalues[b, oh, ow, oc] * weights[oc, kh, kw, ic]
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