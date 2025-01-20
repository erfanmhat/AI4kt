package io.ai4kt.ai4kt.fibonacci.pandas

import io.ai4kt.ai4kt.fibonacci.numpy.np
import kotlin.math.pow
import kotlin.math.sqrt

class Series(private val data: List<Any?>, private val index: List<Any>? = null) : Iterable<Any?> {

    val values
        get() = np.array(data)

    init {
        if (index != null && index.size != data.size) {
            throw IllegalArgumentException("Index length must match data length.")
        }
    }

    private val lazyData by lazy { data }

    val shape: List<Int>
        get() = listOf(data.size)

    operator fun get(position: Int): Any? {
        return data[position]
    }

    operator fun get(label: Any): Any? {
        if (index == null) throw UnsupportedOperationException("Index is not defined.")
        val position = index.indexOf(label)
        if (position == -1) throw NoSuchElementException("Label not found in index.")
        return data[position]
    }

    fun slice(start: Int, end: Int): Series {
        return Series(data.subList(start, end), index?.subList(start, end))
    }

    /**
     * Returns a subset of the Series based on the given indices.
     *
     * @param indices The indices of the elements to include in the subset.
     * @return A new Series containing the specified elements.
     */
    fun slice(indices: List<Int>): Series {
        val slicedData = indices.map { this.data[it] }
        val slicedIndex = this.index?.let { idx -> indices.map { idx[it] } }
        return Series(slicedData, slicedIndex)
    }

    fun filter(condition: (Any?) -> Boolean): Series {
        val filteredData = data.filter(condition)
        val filteredIndex = index?.let { idx ->
            data.indices.filter { condition(data[it]) }.map { idx[it] }
        }
        return Series(filteredData, filteredIndex)
    }

    operator fun plus(other: Series): Series {
        if (this.data.size != other.data.size) throw IllegalArgumentException("Series lengths must match.")
        val result = this.data.mapIndexed { i, a ->
            when (a) {
                is Number -> when (val b = other.data[i]) {
                    is Number -> (a.toDouble() + b.toDouble())
                    else -> throw UnsupportedOperationException("Unsupported type for addition.")
                }

                else -> throw UnsupportedOperationException("Unsupported type for addition.")
            }
        }
        return Series(result, this.index)
    }

    fun sum(): Double {
        return lazyData.filterIsInstance<Number>().sumOf { it.toDouble() }
    }

    fun mean(): Double {
        return sum() / data.size
    }

    fun min(): Any? {
        return lazyData.filterNotNull().filterIsInstance<Comparable<Any>>().minOrNull()
    }

    fun max(): Any? {
        return lazyData.filterNotNull().filterIsInstance<Comparable<Any>>().maxOrNull()
    }

    fun std(): Double {
        val mean = mean()
        val variance = lazyData.filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
        return sqrt(variance)
    }

    fun get_var(): Double {
        val mean = mean()
        return lazyData.filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
    }

    fun dropna(): Series {
        val nonNullData = data.filterNotNull()
        val nonNullIndex = index?.let { idx ->
            data.indices.filter { data[it] != null }.map { idx[it] }
        }
        return Series(nonNullData, nonNullIndex)
    }

    fun fillna(value: Any?): Series {
        val filledData = data.map { it ?: value }
        return Series(filledData, index)
    }

    fun toList(): List<Any?> {
        return data
    }

    fun toMap(): Map<Any, Any?> {
        if (index == null) throw UnsupportedOperationException("Index is not defined.")
        return index.zip(data).toMap()
    }

    override fun iterator(): Iterator<Any?> {
        TODO("Not yet implemented")
    }

    override fun toString(): String {
        val builder = StringBuilder()
        builder.append("Series:\n")
        data.forEachIndexed { i, value ->
            val label = index?.get(i) ?: i
            builder.append("$label\t$value\n")
        }
        return builder.toString()
    }
}

fun main() {
    val numericData = listOf(1, 2, null, 4, 5)
    val numericIndex = listOf("a", "b", "c", "d", "e")
    val numericSeries = Series(numericData, numericIndex)

    println(numericSeries)
    println("Sum: ${numericSeries.sum()}")
    println("Mean: ${numericSeries.mean()}")
    println("Max: ${numericSeries.max()}")
    println("Min: ${numericSeries.min()}")

    val filteredNumericSeries = numericSeries.filter { it != null && (it as Int) > 2 }
    println(filteredNumericSeries)

    val filledNumericSeries = numericSeries.fillna(0)
    println(filledNumericSeries)

    val stringData = listOf("1", "2", null, "4", "5")
    val stringIndex = listOf("a", "b", "c", "d", "e")
    val stringSeries = Series(stringData, stringIndex)

    println(stringSeries)
    println("Max: ${stringSeries.max()}")
    println("Min: ${stringSeries.min()}")

    val filteredStringSeries = stringSeries.filter { it != null && (it as String) > "2" }
    println(filteredStringSeries)

    val filledStringSeries = stringSeries.fillna("0")
    println(filledStringSeries)
}