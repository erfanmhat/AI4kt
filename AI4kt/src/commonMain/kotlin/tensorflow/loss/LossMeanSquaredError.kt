package tensorflow.loss

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class LossMeanSquaredError : Loss {
    // Compute the forward pass (per-sample loss)
    override fun forward(output: D2Array<Double>, y: D2Array<Double>): D1Array<Double> {
        // Ensure output and y have the same shape
        require(output.shape.contentEquals(y.shape)) {
            "Output and y must have the same shape. Output: ${output.shape.contentToString()}, y: ${y.shape.contentToString()}"
        }

        // Calculate the squared differences
        val diff = (output - y)
        val squaredDifferences = diff * diff

        // Compute the mean squared error for each sample
        val squaredDifferencesSum: D1Array<Double> = mk.math.sum(a = squaredDifferences, axis = 1)
        return squaredDifferencesSum / output.shape[1].toDouble()
    }

    // Calculate the gradient of the mean squared error loss
    override fun backward(output: D2Array<Double>, yTrue: D2Array<Double>): D2Array<Double> {
        // Ensure output and yTrue have the same shape
        require(output.shape.contentEquals(yTrue.shape)) {
            "Output and yTrue must have the same shape. Output: ${output.shape.contentToString()}, yTrue: ${yTrue.shape.contentToString()}"
        }

        // Compute the gradient of the MSE loss
        return (output - yTrue) * (2.0 / output.shape[0].toDouble())
    }
}

fun main() {
    // Example predicted values (output) and true values (y)
    val output = mk.ndarray(
        mk[
            mk[3.0, 3.0, 3.0],
            mk[3.0, 3.0, 3.0]
        ]
    ) // D2Array<Double>
    val y = mk.ndarray(
        mk[
            mk[2.0, 2.0, 2.0],
            mk[2.0, 2.0, 2.0]
        ]
    ) // D2Array<Double>

    // Create an instance of LossMeanSquaredError
    val loss = LossMeanSquaredError()

    // Compute the forward pass (per-sample loss)
    val lossValues = loss.forward(output, y)
    println("Per-sample loss: $lossValues")

    // Compute the backward pass (gradient)
    val gradient = loss.backward(output, y)
    println("Gradient: $gradient")
}