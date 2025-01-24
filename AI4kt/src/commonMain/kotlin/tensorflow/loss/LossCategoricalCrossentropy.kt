package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import io.ai4kt.ai4kt.fibonacci.tensorflow.OneHotEncoding
import io.ai4kt.ai4kt.fibonacci.tensorflow.map2DRowsTo1DArray
import io.ai4kt.ai4kt.fibonacci.tensorflow.map2DRowsTo2DArray
import io.ai4kt.ai4kt.fibonacci.tensorflow.times
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOfFirst
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.mapIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.E
import kotlin.math.log
import kotlin.math.max
import kotlin.math.min

class LossCategoricalCrossentropy {
    fun calculate(output: D2Array<Double>, yTrue: D2Array<Double>): Double {
        val samples = output.shape[0]
        val clippedOutput = output.map { value ->
            max(min(value, 1 - 1e-7), 1e-7)
        }
        val confidences = clippedOutput.map2DRowsTo1DArray { i, row ->
            row[yTrue[i].indexOfFirst { it == 1.0 }]
        }
        val losses = confidences.map { -log(it, E) }
        return losses.sum() / samples
    }

    fun backward(output: D2Array<Double>, yTrue: D2Array<Double>): D2Array<Double> {
        val samples = output.shape[0]
        val nRows = output.shape[0]
        val nCols = output.shape[1]

        // Create a result array to store the gradients
        val gradients = mk.zeros<Double>(nRows, nCols)

        // Iterate over each row
        for (i in 0 until nRows) {
            // Iterate over each column
            for (j in 0 until nCols) {
                // Compute the gradient for this element
                gradients[i, j] = if (yTrue[i, j] == 1.0) {
                    -1.0 / (output[i, j] * samples)
                } else {
                    0.0
                }
            }
        }

        return gradients
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
    val lossValue1D = loss1D.calculate(yPred, oneHotEncoder.transform(yTrue1D))
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
    val lossValue2D = loss2D.calculate(yPred, yTrue2D)
    println("Loss (2D yTrue): $lossValue2D")
}