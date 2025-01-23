package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

open class Loss {
    fun calculate(output: D2Array<Double>, y: D2Array<Double>): Double {
        val sampleLosses = forward(output, y)
        return mk.math.sum(sampleLosses) / sampleLosses.shape[0]
    }

    open fun forward(output: D2Array<Double>, y: D2Array<Double>): D1Array<Double> {
        throw NotImplementedError("Forward method must be implemented in the subclass")
    }
}