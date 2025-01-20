package io.ai4kt.ai4kt.fibonacci.numpy

object np {
    /**
     * Creates an ndarray from a List<T?>.
     *
     * @param p_object The input list.
     * @return An ndarray containing the elements of the list.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : Any> array(p_object: List<T?>): ndarray<T> {
        if (p_object.isEmpty()) return ndarray(shape = listOf(0), initialValue = null as T)

        val containsNull = p_object.any { it == null }
        return if (containsNull) {
            val result = ndarray<T?>(shape = listOf(p_object.size), initialValue = p_object[0])
            for ((index, item) in p_object.withIndex()) {
                result[index] = item
            }
            result as ndarray<T>
        } else {
            val nonNullList = p_object.mapNotNull { it }
            val result: ndarray<T> = ndarray(shape = listOf(nonNullList.size), initialValue = nonNullList[0])
            for ((index, item) in p_object.withIndex()) {
                result[index] = item!!
            }
            return result
        }
    }

    fun <T> isnan(x: ndarray<T?>): ndarray<Boolean> {
        val result = ndarray(shape = x.shape, initialValue = false)
        for (index in 0..x.size) {
            if (x[index].data[0] == null) {
                result[index] = true
            }
        }
        return result
    }

    fun any(a: ndarray<Boolean>): Boolean {
        for (item in a.data) {
            if (item != null && item as Boolean) {
                return true
            }
        }
        return false
    }

    fun all(a: ndarray<Boolean?>): Boolean {
        for (item in a.data) {
            if (item != null && !(item as Boolean)) {
                return false
            }
        }
        return true
    }
}

/**
 * Converts a List of Lists or a List of ndarrays to an ndarray.
 *
 * @param T The type of elements in the list (e.g., Double, Int).
 * @return An ndarray containing the elements of the list.
 */
@Suppress("UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
inline fun <reified T : Number> List<*>.to_ndarray(): ndarray<T> {
    // بررسی اینکه لیست خالی نباشد
    if (this.isEmpty()) {
        return ndarray(
            shape = listOf(0, 0), initialValue = when (T::class) {
                Double::class -> 0.0 as T
                Int::class -> 0 as T
                else -> throw IllegalArgumentException("Unsupported type")
            }
        )
    }

    // بررسی نوع عناصر لیست
    val isListOfLists = this[0] is List<*>
    val isListOfNdarray = this[0] is ndarray<*>

    if (!isListOfLists && !isListOfNdarray) {
        throw IllegalArgumentException("Input must be a List of Lists or a List of ndarrays")
    }

    // محاسبه ابعاد
    val numRows = this.size
    val numCols = when {
        isListOfLists -> (this[0] as List<*>).size
        isListOfNdarray -> (this[0] as ndarray<*>).size
        else -> throw IllegalArgumentException("Invalid input type")
    }

    // بررسی یکسان بودن طول سطرها
    for (row in this) {
        val rowSize = when {
            isListOfLists -> (row as List<*>).size
            isListOfNdarray -> (row as ndarray<*>).size
            else -> throw IllegalArgumentException("Invalid input type")
        }
        if (rowSize != numCols) {
            throw IllegalArgumentException("All rows must have the same length")
        }
    }

    // ایجاد ndarray با ابعاد مناسب
    val result = ndarray<T>(
        shape = listOf(numRows, numCols), initialValue = when (T::class) {
            Double::class -> 0.0 as T
            Int::class -> 0 as T
            else -> throw IllegalArgumentException("Unsupported type")
        }
    )

    // پر کردن ndarray با مقادیر
    for (i in 0 until numRows) {
        for (j in 0 until numCols) {
            result[i, j] = when {
                isListOfLists -> (this[i] as List<T>)[j]
                isListOfNdarray -> (this[i] as ndarray<T>)[j].data[0] as T
                else -> throw IllegalArgumentException("Invalid input type")
            } as T
        }
    }

    return result
}