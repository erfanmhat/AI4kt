package io.ai4kt.ai4kt.fibonacci.tensorflow

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

// Extension function to sum two D2Arrays
fun D2Array<Double>.plusD2Array(other: D2Array<Double>): D2Array<Double> {
    // Check if the arrays have the same shape
    require(this.shape.contentEquals(other.shape)) { "Arrays must have the same shape" }

    // Create a new array to store the result
    val result = mk.zeros<Double>(this.shape[0], this.shape[1])

    // Perform element-wise addition
    for (i in 0 until this.shape[0]) {
        for (j in 0 until this.shape[1]) {
            result[i, j] = this[i, j] + other[i, j]
        }
    }

    return result
}

// Extension function to sum a D2Array and a D1Array (broadcasting)
fun D2Array<Double>.plusD1Array(other: D1Array<Double>): D2Array<Double> {
    // Check if the number of columns in the D2Array matches the size of the D1Array
    require(this.shape[1] == other.size) { "D1Array size must match the number of columns in D2Array" }

    // Create a new array to store the result
    val result = mk.zeros<Double>(this.shape[0], this.shape[1])

    // Perform element-wise addition with broadcasting
    for (i in 0 until this.shape[0]) {
        for (j in 0 until this.shape[1]) {
            result[i, j] = this[i, j] + other[j]
        }
    }

    return result
}