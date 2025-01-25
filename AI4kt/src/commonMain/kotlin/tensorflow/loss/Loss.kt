package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

interface Loss {
    fun forward(output: D2Array<Double>, y: D2Array<Double>): D1Array<Double>
    fun backward(output: D2Array<Double>, yTrue: D2Array<Double>): D2Array<Double>
}