package io.ai4kt.ai4kt.fibonacci.sklearn

import io.ai4kt.ai4kt.fibonacci.numpy.ndarray

data class DataSet(
    val X_train: ndarray<Double>,
    val X_test: ndarray<Double>,
    val y_train: ndarray<Int>,
    val y_test: ndarray<Int>
)
