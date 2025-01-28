package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.*

class InputLayer(n_inputs: Int) : Layer {
    val nInputs: Int = n_inputs

    // The forward pass simply returns the input as-is
    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(inputs.shape[1] == nInputs) {
            "Input shape mismatch. Expected $nInputs features, but got ${inputs.shape[1]}."
        }
        return inputs
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        throw Exception("backward not supported for InputLayer")
    }
}