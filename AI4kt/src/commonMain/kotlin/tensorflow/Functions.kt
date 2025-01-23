package io.ai4kt.ai4kt.fibonacci.tensorflow

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

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

fun D2Array<Double>.mapIndexedD2ToD1Array(transform: (Int, D1Array<Double>) -> Double): D1Array<Double> {
    val result = mk.zeros<Double>(this.shape[0])
    for (i in 0 until this.shape[0]) {
        result[i] = transform(i, this[i] as D1Array<Double>)
    }
    return result
}

fun D2Array<Double>.mapIndexedD2ToD2Array(transform: (Int, Int, Double) -> Double): D2Array<Double> {
    val result = mk.zeros<Double>(this.shape[0], this.shape[1])
    for (i in 0 until this.shape[0]) {
        for (j in 0 until this.shape[1]) {
            result[i, j] = transform(i, j, this[i, j])
        }
    }
    return result
}

// Extension function for element-wise multiplication
operator fun D2Array<Double>.times(other: D2Array<Double>): D2Array<Double> {
    require(this.shape.contentEquals(other.shape)) { "Arrays must have the same shape A:${this.shape.contentToString()}, B:${other.shape.contentToString()}" }
    return this.mapIndexedD2ToD2Array { i, j, value ->
        value * other[i, j]
    }
}

fun <T> D1Array<T>.withIndex(): List<IndexedValue<T>> {
    return this.toList().withIndex().toList()
}

fun D2Array<Double>.mapD2ToD1Array(transform: (D1Array<Double>) -> Int): D1Array<Int> {
    val result = mk.zeros<Int>(this.shape[0])
    for (i in 0 until this.shape[0]) {
        result[i] = transform(this[i] as D1Array<Double>)
    }
    return result
}