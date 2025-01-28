package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import io.ai4kt.ai4kt.fibonacci.tensorflow.times
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class ReLU : Activation {
    private lateinit var inputs: NDArray<Double, *>

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        this.inputs = inputs
        return inputs.map { if (it > 0) it else 0.0 }
    }

    override fun backward(dvalues: NDArray<Double, *>, inputs: NDArray<Double, *>): NDArray<Double, *> {
        require(dvalues.shape.contentEquals(inputs.shape)) {
            "dvalues and inputs must have same shape: " +
                    "dvalues.shape=${dvalues.shape.contentToString()}, " +
                    "inputs.shape=${inputs.shape.contentToString()}"
        }
        if (dvalues.shape.size == 2) {
            return (dvalues as D2Array<Double>) * (inputs.map { if (it > 0) 1.0 else 0.0 } as D2Array<Double>)
        } else if (dvalues.shape.size == 4) {
            return (dvalues as D4Array<Double>) * (inputs.map { if (it > 0) 1.0 else 0.0 } as D4Array<Double>)
        }
        throw Exception("this shape not supported in ReLU. shape${dvalues.shape.contentToString()}")
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