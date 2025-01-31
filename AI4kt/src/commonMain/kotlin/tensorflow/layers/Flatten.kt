package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class Flatten(
    inputShape: IntArray
) : Layer {

    var inputShape = intArrayOf()
    var outputShape = intArrayOf()

    init {
        this.inputShape = inputShape
        var flattenedSize = 1
        for (i in 1..<inputShape.size) {
            flattenedSize *= inputShape[i]
        }
        this.outputShape = intArrayOf(inputShape[0], flattenedSize)
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        inputShape = inputs.shape
        var flattenedSize = 1
        for (i in 1..<inputs.shape.size) {
            flattenedSize *= inputs.shape[i]
        }
        return inputs.reshape(inputs.shape[0], flattenedSize)
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        // Reshape the gradients back to the original input shape
        return when (inputShape.size) {
            1 -> dvalues.reshape(dvalues.shape[0], inputShape[0])
            2 -> dvalues.reshape(dvalues.shape[0], inputShape[0], inputShape[1])
            3 -> dvalues.reshape(dvalues.shape[0], inputShape[0], inputShape[1], inputShape[2])
            4 -> dvalues.reshape(dvalues.shape[0], inputShape[0], inputShape[1], inputShape[2], inputShape[3])
            else -> dvalues.reshape(dvalues.shape[0], inputShape[0])
        }
    }
}
