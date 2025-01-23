package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.exp

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import kotlin.math.exp

class ActivationSoftmax {
    fun forward(inputs: D2Array<Double>): D2Array<Double> {
        // Create a new array to store the result
        val output = mk.zeros<Double>(inputs.shape[0], inputs.shape[1])

        // Apply Softmax to each row (batch) independently
        for (i in 0 until inputs.shape[0]) { // Iterate over rows (batch size)
            // Find the maximum value in the row to improve numerical stability
            val maxVal = inputs[i].max() ?: 0.0

            // Compute the exponential of each element in the row, adjusted by the max value
            val expValues = inputs[i].map { exp(it - maxVal) }

            // Compute the sum of exponentials for the row
            val sumExp = expValues.sum()

            // Normalize by dividing each exponential value by the sum
            for (j in 0 until inputs.shape[1]) { // Iterate over columns (features)
                output[i, j] = expValues[j] / sumExp
            }
        }

        return output
    }
}

fun main() {
    // Create a 2D input array (batch of logits)
    val inputs = mk.ndarray(
        listOf(
            listOf(1000.0, 1001.0, 1002.0), // Large numbers
            listOf(-1000.0, -1001.0, -1002.0), // Very small numbers
            listOf(1.0, 2.0, 3.0) // Normal numbers
        )
    )

    // Create an instance of ActivationSoftmax
    val softmax = ActivationSoftmax()

    // Apply the Softmax activation function
    val output = softmax.forward(inputs)

    // Print the result
    println(output)
}