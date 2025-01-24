package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import io.ai4kt.ai4kt.fibonacci.tensorflow.times
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map

class ReLU : Activation {
    private lateinit var inputs: D2Array<Double>

    override fun forward(inputs: D2Array<Double>): D2Array<Double> {
        this.inputs = inputs
        return inputs.map { if (it > 0) it else 0.0 }
    }

    override fun backward(dvalues: D2Array<Double>, inputs: D2Array<Double>): D2Array<Double> {
        return dvalues * inputs.map { if (it > 0) 1.0 else 0.0 }
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
    val relu = ReLU()

    // Apply the ReLU activation function
    val output = relu.forward(inputs)

    // Print the result
    println(output)
}