package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import io.ai4kt.ai4kt.fibonacci.tensorflow.OneHotEncoding
import io.ai4kt.ai4kt.fibonacci.tensorflow.times
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

class LossCategoricalCrossentropy : Loss {
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

    // Calculate the gradient of the categorical cross-entropy loss
    override fun backward(output: D2Array<Double>, yTrue: D2Array<Double>): D2Array<Double> {
        val epsilon = 1e-7
        // Clip output to avoid division by zero
        val clippedOutput = output.map { maxOf(it, epsilon) }
        // Compute the gradient
        return (clippedOutput - yTrue) / output.shape[0].toDouble()
    }
}

fun main() {
    // Create predictions (yPred) and true labels (yTrue)
    val yPred = mk.ndarray(
        listOf(
            listOf(0.05, 0.95, 0.00), // Sample 1
            listOf(0.80, 0.10, 0.10), // Sample 2
            listOf(0.02, 0.00, 0.98)  // Sample 3
        )
    )

    // Case 1: yTrue is 1D (class indices)
    val yTrue1D = mk.ndarray(mk[1, 0, 2]) // Correct classes are 1, 1, 1

    val oneHotEncoder = OneHotEncoding()

    val loss1D = LossCategoricalCrossentropy()
    val lossValue1D = loss1D.forward(yPred, oneHotEncoder.transform(yTrue1D))
    println("Loss (1D yTrue): $lossValue1D")

    // Case 2: yTrue is 2D (one-hot encoded)
    val yTrue2D = mk.ndarray(
        listOf(
            listOf(0.0, 1.0, 0.0), // Sample 1
            listOf(1.0, 0.0, 0.0), // Sample 2
            listOf(0.0, 0.0, 1.0)  // Sample 3
        )
    )

    val loss2D = LossCategoricalCrossentropy()
    val lossValue2D = loss2D.forward(yPred, yTrue2D)
    println("Loss (2D yTrue): $lossValue2D")
}