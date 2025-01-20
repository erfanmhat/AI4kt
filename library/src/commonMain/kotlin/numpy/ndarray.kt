package io.ai4kt.ai4kt.fibonacci.numpy

class ndarray<T : Any?>(
    val shape: List<Int>,
    initialValue: T? = null
) : Iterable<ndarray<T>> {
    var data: Array<Any?>

    val size: Int
        get() = shape.fold(1) { acc, dim -> acc * dim }

    init {
        data = Array(size) { initialValue }
    }

    private fun getIndex(indices: Array<out Int?>): List<Int> {
        if (indices.size > shape.size) {
            throw IllegalArgumentException("Too many indices for array with shape $shape and indices: ${indices.contentToString()}")
        }

        // Generate all possible combinations of indices
        val indexCombinations = mutableListOf<IntArray>()
        generateIndexCombinations(indices, IntArray(shape.size), 0, indexCombinations)

        // Convert each combination to a flat index
        return indexCombinations.map { combination ->
            var index = 0
            var stride = 1
            for (i in shape.size - 1 downTo 0) {
                index += combination[i] * stride
                stride *= shape[i]
            }
            index
        }
    }

    private fun generateIndexCombinations(
        indices: Array<out Int?>,
        currentCombination: IntArray,
        depth: Int,
        result: MutableList<IntArray>
    ) {
        if (depth == shape.size) {
            // Base case: add the current combination to the result
            result.add(currentCombination.copyOf())
            return
        }

        if (depth < indices.size && indices[depth] != null) {
            // If the index is specified, use it
            currentCombination[depth] = indices[depth]!!
            generateIndexCombinations(indices, currentCombination, depth + 1, result)
        } else {
            // If the index is null, iterate over all possible values for this dimension
            for (i in 0 until shape[depth]) {
                currentCombination[depth] = i
                generateIndexCombinations(indices, currentCombination, depth + 1, result)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    operator fun get(vararg indices: Int?): ndarray<T> {
        if (indices.size < shape.size) {
            // Handle partial indexing (return a sub-array)
            return getSubArray(indices)
        }
        val x = mutableListOf<T>()
        for (index in getIndex(indices)) {
            x.add(data[index] as T)
        }
        return np.array(x.toList() as List<Any>) as ndarray<T>
    }

    // todo fix this function to return ndarray<T>
    operator fun get(indices: List<Int>): List<ndarray<T>> {
        return indices.map { this[it] }
    }

    operator fun set(vararg indices: Int?, value: T) {
        for (index in getIndex(indices)) {
            data[index] = value
        }
    }

    private fun getSubArray(indices: Array<out Int?>): ndarray<T> {
        val newShape = shape.subList(indices.size, shape.size)
        val newSize = newShape.fold(1) { acc, dim -> acc * dim }
        val newData = Array(newSize) { null as Any? }

        val startIndex = getIndex(indices)[0]
        for (i in 0 until newSize) {
            newData[i] = data[startIndex + i]
        }

        return ndarray<T>(newShape).apply {
            this.data = newData
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun iterator(): Iterator<ndarray<T>> {
        return object : Iterator<ndarray<T>> {
            private var currentIndex = 0

            override fun hasNext(): Boolean = currentIndex < data.size

            override fun next(): ndarray<T> {
                if (!hasNext()) throw NoSuchElementException("No more elements in the array")
                return ndarray(listOf(1), data[currentIndex++] as T)
            }
        }
    }

    override fun toString(): String {
        fun buildString(currentIndices: Array<Int>, depth: Int): String {
            if (depth == shape.size - 1) {
                // Last dimension: print the elements
                val result = StringBuilder()
                result.append("[")
                for (i in 0 until shape[depth]) {
                    if (i > 0) result.append(", ")
                    currentIndices[depth] = i
                    result.append(data[getIndex(currentIndices)[0]].toString())
                }
                result.append("]")
                return result.toString()
            } else {
                // Not the last dimension: recurse deeper
                val result = StringBuilder()
                result.append("[")
                for (i in 0 until shape[depth]) {
                    if (i > 0) result.append(", ")
                    currentIndices[depth] = i
                    result.append(buildString(currentIndices, depth + 1))
                }
                result.append("]")
                return result.toString()
            }
        }

        // Start with an array of zeros for the indices
        return buildString(Array(shape.size) { 0 }, 0)
    }
}

fun main() {
    val array = ndarray(listOf(2, 3, 4), 0)

    // Set values using partial indices
    array[0, null, 2] =
        5 // Sets all elements in the second dimension to 5 where the first dimension is 0 and the third dimension is 2
    array[1, 2] =
        10 // Sets all elements in the third dimension to 10 where the first dimension is 1 and the second dimension is 2

    println(array)
    println(array[0, null, 2])
    println(array[1, 2])
}