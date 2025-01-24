package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import io.ai4kt.ai4kt.fibonacci.tensorflow.broadcast
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.exp

class Softmax : Activation {
    private lateinit var output: D2Array<Double>

    override fun forward(inputs: D2Array<Double>): D2Array<Double> {
        // Stabilize by subtracting the max value in each row
        val maxValues: D1Array<Double> = mk.math.max(inputs, axis = 1) // Find max value in each row
        val stabilizedInputs = inputs - maxValues.broadcast(inputs.shape[1]) // Subtract max value from each row

        // Compute exp of stabilized inputs
        val expValues = stabilizedInputs.map { exp(it) }

        // Calculate probabilities
        val sumExpValues: D1Array<Double> = mk.math.sum(expValues, axis = 1) // Sum along axis 1
        val broadcastedSumExpValues = sumExpValues.broadcast(expValues.shape[1]) // Shape [M, N]
        val probabilities = expValues / broadcastedSumExpValues // Element-wise division

        this.output = probabilities
        return probabilities
    }

    override fun backward(dvalues: D2Array<Double>, inputs: D2Array<Double>): D2Array<Double> {
        val dinputs = mk.zeros<Double>(inputs.shape[0], inputs.shape[1])

        for (i in 0 until inputs.shape[0]) {
            val singleOutput = output[i] as D1Array<Double>

            // Create the diagonal matrix manually
            val diagOutput = mk.zeros<Double>(singleOutput.size, singleOutput.size)
            for (j in 0 until singleOutput.size) {
                diagOutput[j, j] = singleOutput[j]
            }

            // Create the Jacobian matrix: diag(singleOutput) - singleOutput^T * singleOutput
            val jacobianMatrix = diagOutput - singleOutput.transpose().broadcast(
                diagOutput.shape[0]
            ) dot singleOutput.broadcast(diagOutput.shape[1])

            // Reshape dvalues[i] to a 2D row vector and compute gradients
            val dvaluesRow = dvalues[i].reshape(1, dvalues[i].size)
            val grad = mk.linalg.dot(dvaluesRow, jacobianMatrix)

            // Ensure the shape of grad[0] matches dinputs[i]
            if (grad[0].size == dinputs[i].size) {
                dinputs[i] = grad[0] // Extract the result (1D array)
            } else {
                // Handle the shape mismatch, e.g., by reshaping or logging an error
                throw IllegalArgumentException("Shape mismatch: grad[0] has size ${grad[0].size}, but dinputs[i] expects size ${dinputs[i].size}")
            }
        }

        return dinputs
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
    val softmax = Softmax()

    // Apply the Softmax activation function
    val output = softmax.forward(inputs)

    // Print the result
    println(output)
}