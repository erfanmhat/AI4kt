package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set

class ActivationReLU {
    fun forward(inputs: D2Array<Double>): D2Array<Double> {
        // Create a new array to store the result
        val output = mk.zeros<Double>(inputs.shape[0], inputs.shape[1])

        // Apply the ReLU function element-wise
        for (i in 0 until inputs.shape[0]) { // Iterate over rows (batch size)
            for (j in 0 until inputs.shape[1]) { // Iterate over columns (features)
                output[i, j] = maxOf(0.0, inputs[i, j])
            }
        }

        return output
    }
}

fun main() {
    // Create a 2D input array (batch of inputs)
    val inputs = mk.ndarray(
        listOf(
            listOf(1.0, -2.0, 3.0),
            listOf(-0.5, 0.0, 2.5),
            listOf(-1.0, -3.0, 0.5)
        )
    )

    // Create an instance of ActivationReLU
    val relu = ActivationReLU()

    // Apply the ReLU activation function
    val output = relu.forward(inputs)

    // Print the result
    println(output)
}