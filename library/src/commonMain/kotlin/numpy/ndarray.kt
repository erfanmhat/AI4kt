package io.ai4kt.ai4kt.fibonacci.numpy

import kotlin.math.*

class ndarray<T : Any?>(
    val shape: List<Int>,
    initialValue: T? = null
) : Iterable<ndarray<T>> {
    var data: Array<Any?>

    private val strides: IntArray

    val size: Int = shape.fold(1) { acc, dim -> acc * dim }

    init {
        data = Array(size) { initialValue }
        strides = computeStrides()
    }

    // Precompute strides for efficient flat index calculation
    private fun computeStrides(): IntArray {
        val strides = IntArray(shape.size)
        var stride = 1
        for (i in shape.size - 1 downTo 0) {
            strides[i] = stride
            stride *= shape[i]
        }
        return strides
    }

    // Compute flat index using precomputed strides
    private fun getFlatIndex(indices: IntArray): Int {
        var index = 0
        for (i in indices.indices) {
            index += indices[i] * strides[i]
        }
        return index
    }

    // Optimized set function using iteration instead of recursion
    operator fun set(vararg indices: Int?, value: T) {
        val dims = shape.size
        val starts = IntArray(dims)
        val ends = IntArray(dims)
        for (dim in 0 until dims) {
            if (dim < indices.size && indices[dim] != null) {
                starts[dim] = indices[dim]!!
                ends[dim] = indices[dim]!! + 1
            } else {
                starts[dim] = 0
                ends[dim] = shape[dim]
            }
        }

        val currentIndices = starts.copyOf()
        while (true) {
            val flatIndex = getFlatIndex(currentIndices)
            data[flatIndex] = value

            // Increment indices
            var dim = dims - 1
            while (dim >= 0) {
                currentIndices[dim]++
                if (currentIndices[dim] < ends[dim]) {
                    break
                } else {
                    currentIndices[dim] = starts[dim]
                    dim--
                }
            }
            if (dim < 0) break
        }
    }

    // Optimized get function using iteration
    @Suppress("UNCHECKED_CAST")
    operator fun get(vararg indices: Int?): ndarray<T> {
        if (indices.size < shape.size) {
            // Handle partial indexing (return a sub-array)
            return getSubArray(indices)
        }

        val dims = shape.size
        val starts = IntArray(dims)
        val ends = IntArray(dims)
        for (dim in 0 until dims) {
            if (indices[dim] != null) {
                starts[dim] = indices[dim]!!
                ends[dim] = indices[dim]!! + 1
            } else {
                starts[dim] = 0
                ends[dim] = shape[dim]
            }
        }

        val resultData = mutableListOf<T>()
        val currentIndices = starts.copyOf()
        while (true) {
            val flatIndex = getFlatIndex(currentIndices)
            resultData.add(data[flatIndex] as T)

            // Increment indices
            var dim = dims - 1
            while (dim >= 0) {
                currentIndices[dim]++
                if (currentIndices[dim] < ends[dim]) {
                    break
                } else {
                    currentIndices[dim] = starts[dim]
                    dim--
                }
            }
            if (dim < 0) break
        }

        // Create a new ndarray with the collected data
        return ndarray<T>(listOf(resultData.size)).apply {
            this.data = resultData.toTypedArray<Any?>()
        }
    }

    // Optimized getSubArray function
    private fun getSubArray(indices: Array<out Int?>): ndarray<T> {
        val newShape = shape.subList(indices.size, shape.size)
        val newSize = newShape.fold(1) { acc, dim -> acc * dim }
        val newData = Array<Any?>(newSize) { null }

        val dims = shape.size
        val starts = IntArray(dims)
        val ends = IntArray(dims)
        for (dim in 0 until dims) {
            if (dim < indices.size && indices[dim] != null) {
                starts[dim] = indices[dim]!!
                ends[dim] = indices[dim]!! + 1
            } else {
                starts[dim] = 0
                ends[dim] = shape[dim]
            }
        }

        val currentIndices = starts.copyOf()
        var idx = 0
        while (true) {
            val flatIndex = getFlatIndex(currentIndices)
            newData[idx++] = data[flatIndex]

            // Increment indices
            var dim = dims - 1
            while (dim >= 0) {
                currentIndices[dim]++
                if (currentIndices[dim] < ends[dim]) {
                    break
                } else {
                    currentIndices[dim] = starts[dim]
                    dim--
                }
            }
            if (dim < 0) break
        }

        return ndarray<T>(newShape).apply {
            this.data = newData
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun iterator(): Iterator<ndarray<T>> {
        return data.map { ndarray(listOf(1), it as T) }.iterator()
    }

    // Optimized toString function
    override fun toString(): String {
        val result = StringBuilder()
        fun buildString(currentIndices: IntArray, depth: Int) {
            if (depth == shape.size) {
                result.append(data[getFlatIndex(currentIndices)].toString())
                return
            }

            result.append("[")
            for (i in 0 until shape[depth]) {
                if (i > 0) result.append(", ")
                currentIndices[depth] = i
                buildString(currentIndices, depth + 1)
            }
            result.append("]")
        }

        buildString(IntArray(shape.size), 0)
        return result.toString()
    }
}

// Usage example
fun main() {
    val array = ndarray(listOf(2, 3, 4), 0)

    // Set values using partial indices
    array[0, null, 2] = 5 // Set elements where first dim is 0 and third dim is 2
    array[1, 2] = 10     // Set elements where first dim is 1 and second dim is 2

    println(array)
    println(array[0, null, 2])
    println(array[1, 2])
}
