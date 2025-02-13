package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class Input(vararg inputShape: Int) : Layer {
    val inputShape: IntArray = inputShape

    // The forward pass simply returns the input as-is
    override suspend fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(
            inputs.shape.toList().subList(1, inputs.shape.size) ==
                    inputShape.toList()
        ) {
            "Input shape mismatch. Expected ${inputShape.joinToString(", ")}, but got ${inputs.shape.joinToString(", ")}."
        }
        return inputs
    }

    override suspend fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        throw Exception("backward not supported for InputLayer")
    }
}