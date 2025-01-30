package tensorflow

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

// Extension function to sum a D2Array and a D1Array (broadcasting)
fun D2Array<Double>.D2PlusD1Array(other: D1Array<Double>): D2Array<Double> {
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
                (0 until this.shape[0]).maxByOrNull { row -> this[row, col] }
                    ?: throw NoSuchElementException("Column is empty.")
            }
        }

        1 -> {
            // Find the index of the maximum value along each row (axis 1)
            mk.d1array(this.shape[0]) { row ->
                (0 until this.shape[1]).maxByOrNull { col -> this[row, col] }
                    ?: throw NoSuchElementException("Row is empty.")
            }
        }

        else -> throw IllegalArgumentException("Axis must be null, 0, or 1.")
    }
}

/**
 * Finds the 2D indices (row, column) of the maximum value in a 2D array.
 *
 * @param array The 2D array to search.
 * @return A Pair containing the row and column indices of the maximum value.
 */
fun findMaxIndex(array: D2Array<Double>): Pair<Int, Int> {
    var maxValue = array[0, 0]
    var maxRow = 0
    var maxCol = 0

    // Iterate through the array to find the maximum value and its indices
    for (i in 0 until array.shape[0]) {
        for (j in 0 until array.shape[1]) {
            if (array[i, j] > maxValue) {
                maxValue = array[i, j]
                maxRow = i
                maxCol = j
            }
        }
    }

    return Pair(maxRow, maxCol)
}

/**
 * Compares two NDArrays for equality by checking their shapes and content.
 *
 * @param other The NDArray to compare with.
 * @return `true` if the shapes and content are equal, `false` otherwise.
 */
fun <T : Number> NDArray<T, *>.contentEquals(other: NDArray<T, *>): Boolean {
    // Check if shapes are equal
    if (!this.shape.contentEquals(other.shape)) {
        return false
    }

    when (this.shape.size) {
        1 -> for (i in 0..<this.shape[0]) {
            if ((this as D1Array<T>)[i] != (other as D1Array<T>)[i]) {
                return false
            }
        }

        2 -> for (i in 0..<this.shape[0]) {
            for (j in 0..<this.shape[1]) {
                if ((this as D2Array<T>)[i, j] != (other as D2Array<T>)[i, j]) {
                    return false
                }
            }
        }

        3 -> for (i in 0..<this.shape[0]) {
            for (j in 0..<this.shape[1]) {
                for (k in 0..<this.shape[2]) {
                    if ((this as D3Array<T>)[i, j, k] != (other as D3Array<T>)[i, j, k]) {
                        return false
                    }
                }
            }
        }

        4 -> for (i in 0..<this.shape[0]) {
            for (j in 0..<this.shape[1]) {
                for (k in 0..<this.shape[2]) {
                    for (l in 0..<this.shape[3]) {
                        if ((this as D4Array<T>)[i, j, k, l] != (other as D4Array<T>)[i, j, k, l]) {
                            return false
                        }
                    }
                }
            }
        }
    }

    return true
}

operator fun <E> List<E>.get(intRange: IntRange): List<E> {
    val result = mutableListOf<E>()
    for (index in intRange) {
        result.add(this[index])
    }
    return result
}

// Extension function to sum a D4Array and a D1Array (broadcasting)
fun D4Array<Double>.D4PlusD1Array(other: D1Array<Double>): D4Array<Double> {
    // Check if the number of channels in the D4Array matches the size of the D1Array
    require(this.shape[3] == other.size) {
        "D1Array size must match the number of channels in D4Array " +
                "D4Array.shape: ${this.shape.contentToString()} " +
                "D1Array.shape: ${other.shape.contentToString()}"
    }

    // Create a new array to store the result
    val result = mk.zeros<Double>(this.shape[0], this.shape[1], this.shape[2], this.shape[3])

    // Perform element-wise addition with broadcasting
    for (i in 0 until this.shape[0]) { // Batch dimension
        for (j in 0 until this.shape[1]) { // Channel dimension
            for (k in 0 until this.shape[2]) { // Height dimension
                for (l in 0 until this.shape[3]) { // Width dimension
                    result[i, j, k, l] = this[i, j, k, l] + other[l] // Corrected indexing
                }
            }
        }
    }

    return result
}


operator fun NDArray<Double, *>.get(intRange: IntRange): NDArray<Double, *> {
    // Ensure the range is valid for the first dimension of the array
    require(intRange.first >= 0 && intRange.last < this.shape[0]) {
        "Range $intRange is out of bounds for dimension 0 with size ${this.shape[0]}"
    }

    // Create a new shape for the sliced array
    val newShape = this.shape.toMutableList().apply {
        this[0] = intRange.count() // Update the size of the first dimension
    }.toIntArray()

    // Create a new array to store the sliced data
    return when (newShape.size) {
        1 -> {
            val slicedArray = mk.zeros<Double>(newShape[0])
            for (i in intRange.first..intRange.last) {
                slicedArray[i - intRange.first] = (this as D1Array<Double>)[i]
            }
            slicedArray
        }

        2 -> {
            val slicedArray = mk.zeros<Double>(newShape[0], newShape[1])
            for (i in intRange.first..intRange.last) {
                for (j in 0..<this.shape[1]) {
                    slicedArray[i - intRange.first, j] = (this as D2Array<Double>)[i, j]
                }
            }
            slicedArray
        }

        3 -> {
            val slicedArray = mk.zeros<Double>(newShape[0], newShape[1], newShape[2])
            for (i in intRange.first..intRange.last) {
                for (j in 0..<this.shape[1]) {
                    for (k in 0..<this.shape[2]) {
                        slicedArray[i - intRange.first, j, k] = (this as D3Array<Double>)[i, j, k]
                    }
                }
            }
            slicedArray
        }

        4 -> {
            val slicedArray = mk.zeros<Double>(newShape[0], newShape[1], newShape[2], newShape[3])
            for (i in intRange.first..intRange.last) {
                for (j in 0..<this.shape[1]) {
                    for (k in 0..<this.shape[2]) {
                        for (l in 0..<this.shape[3]) {
                            slicedArray[i - intRange.first, j, k, l] = (this as D4Array<Double>)[i, j, k, l]
                        }
                    }
                }
            }
            slicedArray
        }

        else -> throw Exception("unsupported dimension shape: ${newShape.contentToString()}")
    }
}

fun main() {
    // Create a 4D array (batchSize = 2, channels = 3, height = 2, width = 2)
    val d4Array = mk.ndarray(
        listOf(
            listOf(
                listOf(listOf(1.0, 2.0), listOf(3.0, 4.0)),
                listOf(listOf(5.0, 6.0), listOf(7.0, 8.0)),
                listOf(listOf(9.0, 10.0), listOf(11.0, 12.0))
            ),
            listOf(
                listOf(listOf(13.0, 14.0), listOf(15.0, 16.0)),
                listOf(listOf(17.0, 18.0), listOf(19.0, 20.0)),
                listOf(listOf(21.0, 22.0), listOf(23.0, 24.0))
            )
        )
    )

    // Create a 1D array (size = 3, matching the number of channels)
    val d1Array = mk.ndarray(listOf(1.0, 2.0, 3.0))

    // Add the 1D array to the 4D array
    val result = d4Array.D4PlusD1Array(d1Array)

    // Print the result
    println(result)
}

//fun main() {
//    val data = mk.d2array(3, 3) { (it % 3 + it / 3).toDouble() } // Example 2D array
//    println("Data:")
//    println(data)
//
//    // No axis specified (flattened array)
//    val argmaxFlattened = data.argmax()
//    println("\nIndex of max value in flattened array: $argmaxFlattened")
//
//    // Axis 0 (columns)
//    val argmaxAxis0 = data.argmax(axis = 0)
//    println("\nIndices of max values along axis 0 (columns):")
//    println(argmaxAxis0)
//
//    // Axis 1 (rows)
//    val argmaxAxis1 = data.argmax(axis = 1)
//    println("\nIndices of max values along axis 1 (rows):")
//    println(argmaxAxis1)
//}

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