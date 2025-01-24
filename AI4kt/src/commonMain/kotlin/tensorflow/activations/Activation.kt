package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

interface Activation {
    fun forward(inputs: D2Array<Double>): D2Array<Double>
    fun backward(dvalues: D2Array<Double>, inputs: D2Array<Double>): D2Array<Double>
}