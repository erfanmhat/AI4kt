package io.ai4kt.ai4kt.fibonacci.sklearn

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array


data class DataSet(
    val X_train: D2Array<Double>,
    val X_test: D2Array<Double>,
    val y_train: D1Array<Int>,
    val y_test: D1Array<Int>
)
