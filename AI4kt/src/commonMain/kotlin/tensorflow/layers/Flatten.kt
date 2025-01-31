package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class Flatten(
    val batchSize: Int,
    inputShape: IntArray
) : Layer {

    var inputShape: IntArray
    var outputShape: IntArray
    var flattenedSize: Int = 0

    init {
        this.inputShape = inputShape

        flattenedSize = 1
        for (i in inputShape.indices) {
            flattenedSize *= inputShape[i]
        }

        outputShape = intArrayOf(batchSize, flattenedSize)
    }

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        return try {
            inputs.reshape(batchSize, flattenedSize)
        } catch (e: Exception) {
            try {
                if (inputs.size / flattenedSize < batchSize) {
                    inputs.reshape(inputs.size / flattenedSize, flattenedSize)
                } else {
                    throw Exception(
                        "can not reshape inputs with shape: ${inputs.shape.contentToString()} " +
                                "to shape: [$batchSize, $flattenedSize]"
                    )
                }
            } catch (e: Exception) {
                throw Exception(
                    "can not reshape inputs with shape: ${inputs.shape.contentToString()} " +
                            "to shape: [$batchSize, $flattenedSize] or " +
                            "[${inputs.size / flattenedSize},$flattenedSize]"
                )
            }
        }
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        // Reshape the gradients back to the original input shape
        return when (inputShape.size) {
            1 -> dvalues.reshape(batchSize, inputShape[0])
            2 -> dvalues.reshape(batchSize, inputShape[0], inputShape[1])
            3 -> dvalues.reshape(batchSize, inputShape[0], inputShape[1], inputShape[2])
            4 -> dvalues.reshape(batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3])
            else -> dvalues.reshape(batchSize, inputShape[0])
        }
    }
}
