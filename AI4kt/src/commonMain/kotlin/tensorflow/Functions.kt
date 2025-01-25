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

/**
 * Broadcasts a 1D array to a 2D array along the specified axis.
 *
 * @param n The size of the new axis to broadcast to.
 * @param axis The axis along which to broadcast (0 or 1).
 * @return A 2D array with the broadcasted values.
 */
fun D1Array<Double>.broadcast(n: Int, axis: Int = 0): D2Array<Double> {
    val m = this.size

    return when (axis) {
        0 -> {
            // Broadcast along axis 0: [1, N] -> [M, N]
            val broadcastedArray = mk.zeros<Double>(n, m) // Shape [N, M]
            for (i in 0 until n) {
                for (j in 0 until m) {
                    broadcastedArray[i, j] = this[j] // Repeat rows
                }
            }
            broadcastedArray
        }

        1 -> {
            // Broadcast along axis 1: [M, 1] -> [M, N]
            val broadcastedArray = mk.zeros<Double>(m, n) // Shape [M, N]
            for (i in 0 until m) {
                for (j in 0 until n) {
                    broadcastedArray[i, j] = this[i] // Repeat columns
                }
            }
            broadcastedArray
        }

        else -> throw IllegalArgumentException("Axis must be 0 or 1.")
    }
}

/**
 * Returns the indices of the maximum values along the specified axis.
 *
 * @param axis The axis along which to find the indices of the maximum values.
 *             If null, the index is computed for the flattened array.
 * @return The indices of the maximum values.
 */
fun D2Array<Double>.argmax(axis: Int? = null): Any {
    return when (axis) {
        null -> {
            // Flatten the array and find the index of the maximum value
            val flattened = this.flatten()
            flattened.indices.maxByOrNull { flattened[it] } ?: throw NoSuchElementException("Array is empty.")
        }
        0 -> {
            // Find the index of the maximum value along each column (axis 0)
            mk.d1array(this.shape[1]) { col ->
                (0 until this.shape[0]).maxByOrNull { row -> this[row, col] } ?: throw NoSuchElementException("Column is empty.")
            }
        }
        1 -> {
            // Find the index of the maximum value along each row (axis 1)
            mk.d1array(this.shape[0]) { row ->
                (0 until this.shape[1]).maxByOrNull { col -> this[row, col] } ?: throw NoSuchElementException("Row is empty.")
            }
        }
        else -> throw IllegalArgumentException("Axis must be null, 0, or 1.")
    }
}

fun main() {
    val data = mk.d2array(3, 3) { (it % 3 + it / 3).toDouble() } // Example 2D array
    println("Data:")
    println(data)

    // No axis specified (flattened array)
    val argmaxFlattened = data.argmax()
    println("\nIndex of max value in flattened array: $argmaxFlattened")

    // Axis 0 (columns)
    val argmaxAxis0 = data.argmax(axis = 0)
    println("\nIndices of max values along axis 0 (columns):")
    println(argmaxAxis0)

    // Axis 1 (rows)
    val argmaxAxis1 = data.argmax(axis = 1)
    println("\nIndices of max values along axis 1 (rows):")
    println(argmaxAxis1)
}

//fun main() {
//    // Create a [M, 1] array
//    val array = mk.ndarray(
//        listOf(1.0, 2.0, 3.0)
//    )
//
//    println("Original array (shape [M, 1]):")
//    println(array)
//
//    // Broadcast to [M, N]
//    val broadcastedArray = array.broadcast(n = 4, axis = 0)
//    println("\nBroadcasted array (shape [M, N]):")
//    println(broadcastedArray)
//
//    // Broadcast to [N, M]
//    val broadcastedArray2 = array.broadcast(n = 4, axis = 1)
//    println("\nBroadcasted array (shape [M, N]):")
//    println(broadcastedArray2)
//}