package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

interface Layer {
    fun forward(inputs: NDArray<Double, *>): NDArray<Double, *>
    fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *>
}