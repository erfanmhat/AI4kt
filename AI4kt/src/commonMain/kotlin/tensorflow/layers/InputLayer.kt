package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class InputLayer(vararg inputShape: Int) : Layer {
    val inputShape: IntArray = inputShape

    // The forward pass simply returns the input as-is
    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(inputs.shape.contentEquals(inputShape)) {
            "Input shape mismatch. Expected ${inputShape.joinToString(", ")}, but got ${inputs.shape.joinToString(", ")}."
        }
        return inputs
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        throw Exception("backward not supported for InputLayer")
    }
}