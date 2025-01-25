package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

interface Layer {
    fun forward(inputs: D2Array<Double>): D2Array<Double>
    fun backward(dvalues: D2Array<Double>): D2Array<Double>
}