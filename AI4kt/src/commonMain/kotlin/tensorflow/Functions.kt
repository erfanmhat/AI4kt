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

// Extension function for element-wise multiplication
operator fun D2Array<Double>.times(other: D2Array<Double>): D2Array<Double> {
    require(this.shape.contentEquals(other.shape)) { "Arrays must have the same shape A:${this.shape.contentToString()}, B:${other.shape.contentToString()}" }
    return this.map2DElementsWithIndices { i, j, value ->
        value * other[i, j]
    }
}

fun <T> D1Array<T>.withIndex(): List<IndexedValue<T>> {
    return this.toList().withIndex().toList()
}

inline fun <T : Number, reified R : Number> D2Array<T>.map2DRowsTo1DArray(transform: (Int, D1Array<T>) -> R): D1Array<R> {
    val result = mk.zeros<R>(this.shape[0])
    for (i in 0 until this.shape[0]) {
        result[i] = transform(i, this[i] as D1Array<T>)
    }
    return result
}

inline fun <T : Number, reified R : Number> D2Array<T>.map2DElementsWithIndices(transform: (Int, Int, T) -> R): D2Array<R> {
    val result = mk.zeros<R>(this.shape[0], this.shape[1])
    for (i in 0 until this.shape[0]) {
        for (j in 0 until this.shape[1]) {
            result[i, j] = transform(i, j, this[i, j])
        }
    }
    return result
}

inline fun <T : Number> D2Array<T>.map2DRowsToIntArray(transform: (D1Array<T>) -> Int): D1Array<Int> {
    val result = mk.zeros<Int>(this.shape[0])
    for (i in 0 until this.shape[0]) {
        result[i] = transform(this[i] as D1Array<T>)
    }
    return result
}

inline fun <T : Number, reified R : Number> D2Array<T>.map2DRowsTo2DArray(transform: (D1Array<T>) -> D1Array<R>): D2Array<R> {
    val result = mk.zeros<R>(this.shape[0], this.shape[1])
    for (i in 0 until this.shape[0]) {
        result[i] = transform(this[i] as D1Array<T>)
    }
    return result
}

// Function to broadcast a [M, 1] array to [M, N]
fun D1Array<Double>.broadcast(n: Int): D2Array<Double> {
    val (m, _) = this.shape
    val broadcastedArray = mk.zeros<Double>(m, n) // Create a new array of shape [M, N]

    // Fill the new array by repeating values along the second axis
    for (i in 0 until m) {
        for (j in 0 until n) {
            broadcastedArray[i, j] = this[i] // Repeat the value from [i, 0] across all columns
        }
    }

    return broadcastedArray
}

fun main() {
    // Create a [M, 1] array
    val array = mk.ndarray(
        listOf(1.0, 2.0, 3.0)
    )

    println("Original array (shape [M, 1]):")
    println(array)

    // Broadcast to [M, N]
    val broadcastedArray = array.broadcast(n = 4)
    println("\nBroadcasted array (shape [M, N]):")
    println(broadcastedArray)
}