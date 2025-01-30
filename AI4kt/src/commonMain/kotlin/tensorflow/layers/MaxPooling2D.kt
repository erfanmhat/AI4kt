package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.findMaxIndex

class MaxPooling2D(
    val poolSize: Pair<Int, Int>, // Size of the pooling window (height, width)
    val strides: Pair<Int, Int>, // Strides for sliding the window (height, width)
    val padding: String, // Padding mode: "valid" or "same"
    val inputShape: IntArray // Shape of the input (batchSize, height, width, channels)
) : Layer {
    private var outputShape: IntArray
    private lateinit var inputCache: NDArray<Double, D4> // Cache for storing input during forward pass

    init {
        // Calculate output shape based on input shape, pool size, strides, and padding
        val (batchSize, inputHeight, inputWidth, channels) = inputShape
        val (poolHeight, poolWidth) = poolSize
        val (strideHeight, strideWidth) = strides

        val outputHeight = when (padding) {
            "valid" -> (inputHeight - poolHeight) / strideHeight + 1
            "same" -> (inputHeight + strideHeight - 1) / strideHeight
            else -> throw IllegalArgumentException("Invalid padding mode: $padding")
        }

        val outputWidth = when (padding) {
            "valid" -> (inputWidth - poolWidth) / strideWidth + 1
            "same" -> (inputWidth + strideWidth - 1) / strideWidth
            else -> throw IllegalArgumentException("Invalid padding mode: $padding")
        }

        outputShape = intArrayOf(batchSize, outputHeight, outputWidth, channels)
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        // Cache the input for use in the backward pass
        inputCache = inputs as NDArray<Double, D4>

        val (batchSize, inputHeight, inputWidth, channels) = inputShape
        val (poolHeight, poolWidth) = poolSize
        val (strideHeight, strideWidth) = strides

        // Initialize the output array
        val output = mk.zeros<Double>(outputShape[0], outputShape[1], outputShape[2], outputShape[3])

        // Perform max pooling
        for (b in 0 until batchSize) { // Iterate over batches
            for (c in 0 until channels) { // Iterate over channels
                for (i in 0 until outputShape[1]) { // Iterate over output height
                    for (j in 0 until outputShape[2]) { // Iterate over output width
                        // Calculate the window boundaries
                        val startH = i * strideHeight
                        val endH = startH + poolHeight
                        val startW = j * strideWidth
                        val endW = startW + poolWidth

                        // Extract the window from the input
                        val window = inputs[b, startH until endH, startW until endW, c]

                        // Compute the maximum value in the window
                        output[b, i, j, c] = window.max()!!
                    }
                }
            }
        }

        return output
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        // Initialize the gradient array with the same shape as the input
        val gradients =
            mk.zeros<Double>(inputCache.shape[0], inputCache.shape[1], inputCache.shape[2], inputCache.shape[3])

        val (batchSize, inputHeight, inputWidth, channels) = inputShape
        val (poolHeight, poolWidth) = poolSize
        val (strideHeight, strideWidth) = strides

        // Backpropagate the gradients
        for (b in 0 until batchSize) { // Iterate over batches
            for (c in 0 until channels) { // Iterate over channels
                for (i in 0 until outputShape[1]) { // Iterate over output height
                    for (j in 0 until outputShape[2]) { // Iterate over output width
                        // Calculate the window boundaries
                        val startH = i * strideHeight
                        val endH = startH + poolHeight
                        val startW = j * strideWidth
                        val endW = startW + poolWidth

                        // Extract the window from the cached input
                        val window = inputCache[b, startH until endH, startW until endW, c]

                        // Find the index of the maximum value in the window
                        val maxIndex = findMaxIndex(window as D2Array<Double>)

                        // Compute the position of the maximum value in the input
                        val maxH = startH + maxIndex.first
                        val maxW = startW + maxIndex.second

                        // Propagate the gradient to the position of the maximum value
                        gradients[b, maxH, maxW, c] = dvalues[b, i, j, c]
                    }
                }
            }
        }

        return gradients
    }
}