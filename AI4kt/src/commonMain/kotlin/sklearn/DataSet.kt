package io.ai4kt.ai4kt.fibonacci.sklearn

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array


data class DataSet(
    var X_train: D2Array<Double>,
    var X_test: D2Array<Double>,
    var y_train: D1Array<Int>,
    var y_test: D1Array<Int>
)
