package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.ln

class LossBinaryCrossentropy : Loss {
    // Compute the forward pass (per-sample loss)
    override fun forward(output: D2Array<Double>, y: D2Array<Double>): D1Array<Double> {
        val epsilon = 1e-7
        // Clip output to avoid log(0)
        val clippedOutput = output.map { maxOf(it, epsilon) }
        // Calculate negative log probabilities
        val logProbs = clippedOutput.map { -ln(it) }
        // Compute per-sample loss
        return mk.math.sum(logProbs * y, axis = 1)
    }

    // Calculate the gradient of the binary cross-entropy loss
    override fun backward(output: D2Array<Double>, yTrue: D2Array<Double>): D2Array<Double> {
        val epsilon = 1e-7
        // Clip output to avoid division by zero
        val clippedOutput = output.map { maxOf(it, epsilon) }
        // Compute the gradient
        return (clippedOutput - yTrue) / output.shape[0].toDouble()
    }
}