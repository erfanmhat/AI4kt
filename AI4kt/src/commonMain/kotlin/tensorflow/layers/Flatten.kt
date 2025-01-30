package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class Flatten : Layer {

    lateinit var inputShape: IntArray
    var outputShape: IntArray = intArrayOf()

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        // Store the input shape
        inputShape = inputs.shape
        val batchSize = inputs.shape[0]
        val flattenedSize = inputs.size / batchSize

        // Reshape the inputs to (batchSize, flattenedSize)
        outputShape = intArrayOf(batchSize, flattenedSize)
        return inputs.reshape(batchSize, flattenedSize)
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        // Reshape the gradients back to the original input shape
        return when (inputShape.size) {
            1 -> dvalues.reshape(inputShape[0])
            2 -> dvalues.reshape(inputShape[0], inputShape[1])
            3 -> dvalues.reshape(inputShape[0], inputShape[1], inputShape[2])
            4 -> dvalues.reshape(inputShape[0], inputShape[1], inputShape[2], inputShape[3])
            else -> dvalues.reshape(inputShape[0])
        }
    }
}
