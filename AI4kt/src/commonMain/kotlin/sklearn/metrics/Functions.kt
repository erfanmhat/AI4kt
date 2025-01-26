package io.ai4kt.ai4kt.fibonacci.sklearn.metrics

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun accuracy_score(
    y_test: D1Array<Int>,
    y_pred: D1Array<Int>
): Double {
    var accuracy = 0.0
    for (index in 0..<y_test.shape[0]) {
        if (y_test[index] == y_pred[index]) {
            accuracy += 1
        }
    }
    return accuracy / y_test.shape[0]
}

/**
 * Computes the Mean Squared Error (MSE) between true and predicted values.
 *
 * @param y_true The true target values (D1Array<Double> or D1Array<Int>).
 * @param y_pred The predicted target values (D1Array<Double> or D1Array<Int>).
 * @return The Mean Squared Error as a Double.
 */
fun mean_squared_error(y_true: D1Array<*>, y_pred: D1Array<*>): Double {
    // Ensure y_true and y_pred have the same shape
    require(y_true.shape.contentEquals(y_pred.shape)) {
        "y_true and y_pred must have the same shape. y_true: ${y_true.shape.contentToString()}, y_pred: ${y_pred.shape.contentToString()}"
    }

    // Convert y_true and y_pred to D1Array<Double>
    val yTrueDouble = y_true.toDoubleArray()
    val yPredDouble = y_pred.toDoubleArray()

    // Calculate the squared differences
    val squaredDifferences = yTrueDouble - yPredDouble
    val squaredErrors = squaredDifferences * squaredDifferences

    // Calculate the mean of squared errors
    return squaredErrors.sum() / y_true.size
}

/**
 * Converts a D1Array of any numeric type to a D1Array<Double>.
 *
 * @param array The input array (D1Array<*>).
 * @return A D1Array<Double> containing the converted values.
 */
private fun D1Array<*>.toDoubleArray(): D1Array<Double> {
    return when (this[0]) {
        is Int -> mk.ndarray(this.toList().map { (it as Int).toDouble() })
        is Double -> mk.ndarray(this.toList().map { it as Double })
        else -> throw IllegalArgumentException("Unsupported type: ${this[0]!!::class}")
    }
}