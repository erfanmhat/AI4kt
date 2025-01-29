package tensorflow.activations

import tensorflow.broadcast
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.exp

class Softmax : Activation {
    private lateinit var output: D2Array<Double>

    override fun forward(inputs: NDArray<Double, *>): D2Array<Double> {
        // Stabilize by subtracting the max value in each row
        val maxValues: D1Array<Double> = mk.math.max(inputs, axis = 1) // Find max value in each row
        val stabilizedInputs =
            (inputs as D2Array<Double>) - maxValues.broadcast(
                inputs.shape[1],
                axis = 1
            ) // Subtract max value from each row

        // Compute exp of stabilized inputs
        val expValues = stabilizedInputs.map { exp(it) }

        // Calculate probabilities
        val sumExpValues: D1Array<Double> = mk.math.sum(expValues, axis = 1) // Sum along axis 1
        val broadcastedSumExpValues = sumExpValues.broadcast(expValues.shape[1], axis = 1) // Shape [M, N]
        val probabilities = expValues / broadcastedSumExpValues // Element-wise division

        this.output = probabilities
        return probabilities
    }

    override fun backward(dvalues: NDArray<Double, *>, inputs: NDArray<Double, *>): D2Array<Double> {
        require(dvalues.shape.contentEquals(inputs.shape)) {
            "dvalues and inputs must have same shape: " +
                    "dvalues.shape=${dvalues.shape.contentToString()}, " +
                    "inputs.shape=${inputs.shape.contentToString()}"
        }

        val dinputs = mk.zeros<Double>(inputs.shape[0], inputs.shape[1])

        for (i in 0 until inputs.shape[0]) {
            val singleOutput = output[i] as D1Array<Double>

            // Create the diagonal matrix manually
            val diagOutput = mk.zeros<Double>(singleOutput.size, singleOutput.size)
            for (j in 0 until singleOutput.size) {
                diagOutput[j, j] = singleOutput[j]
            }

            // Compute the outer product: s^T * s
            val outerProduct = singleOutput.reshape(singleOutput.size, 1) dot singleOutput.reshape(1, singleOutput.size)

            // Create the Jacobian matrix: diag(s) - s^T * s
            val jacobianMatrix = diagOutput - outerProduct

            // Reshape dvalues[i] to a 2D row vector and compute gradients
            val dvaluesRow = (dvalues as D2Array<Double>)[i].reshape(1, dvalues[i].size)
            val grad = mk.linalg.dot(dvaluesRow, jacobianMatrix)

            // Ensure the shape of grad matches dinputs[i]
            if (grad.shape[1] == dinputs[i].size) {
                dinputs[i] = grad[0] // Extract the result (1D array)
            } else {
                throw IllegalArgumentException("Shape mismatch: grad[0] has size ${grad.shape[1]}, but dinputs[i] expects size ${dinputs[i].size}")
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