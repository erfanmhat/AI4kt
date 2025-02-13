package tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

interface Layer {
    suspend fun forward(inputs: NDArray<Double, *>): NDArray<Double, *>
    suspend fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *>
}