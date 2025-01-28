package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

interface Activation {
    fun forward(inputs: NDArray<Double, *>): NDArray<Double, *>
    fun backward(dvalues: NDArray<Double, *>, inputs: NDArray<Double, *>): NDArray<Double, *>
}