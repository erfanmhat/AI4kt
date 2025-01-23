package io.ai4kt.ai4kt.fibonacci.sklearn.metrics

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get

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