package io.ai4kt.ai4kt.fibonacci.pandas

import kotlin.math.pow
import kotlin.math.sqrt

class Series(private val data: List<Any?>, private val index: List<Any>? = null) {

    init {
        if (index != null && index.size != data.size) {
            throw IllegalArgumentException("Index length must match data length.")
        }
    }

    // Lazy evaluation support
    private val lazyData by lazy { data }

    val size: Int
        get() = data.size

    // Basic indexing (position-based)
    operator fun get(position: Int): Any? {
        return data[position]
    }

    // Label-based indexing
    operator fun get(label: Any): Any? {
        if (index == null) throw UnsupportedOperationException("Index is not defined.")
        val position = index.indexOf(label)
        if (position == -1) throw NoSuchElementException("Label not found in index.")
        return data[position]
    }

    // Slicing
    fun slice(start: Int, end: Int): Series {
        return Series(data.subList(start, end), index?.subList(start, end))
    }

    // Boolean indexing
    fun filter(condition: (Any?) -> Boolean): Series {
        val filteredData = data.filter(condition)
        val filteredIndex = index?.let { idx ->
            data.indices.filter { condition(data[it]) }.map { idx[it] }
        }
        return Series(filteredData, filteredIndex)
    }

    // Arithmetic operations (element-wise)
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

    // Statistical operations
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

    // Missing data handling
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

    // Conversion
    fun toList(): List<Any?> {
        return data
    }

    fun toMap(): Map<Any, Any?> {
        if (index == null) throw UnsupportedOperationException("Index is not defined.")
        return index.zip(data).toMap()
    }

    // Pretty-printing
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
    // Example with numeric data
    val numericData = listOf(1, 2, null, 4, 5)
    val numericIndex = listOf("a", "b", "c", "d", "e")
    val numericSeries = Series(numericData, numericIndex)

    println(numericSeries) // Pretty-printed output
    println("Sum: ${numericSeries.sum()}") // Sum of non-null values
    println("Mean: ${numericSeries.mean()}") // Mean of non-null values
    println("Max: ${numericSeries.max()}") // Max value
    println("Min: ${numericSeries.min()}") // Min value

    val filteredNumericSeries = numericSeries.filter { it != null && (it as Int) > 2 }
    println(filteredNumericSeries) // Filtered series

    val filledNumericSeries = numericSeries.fillna(0)
    println(filledNumericSeries) // Series with nulls filled

    // Example with string data
    val stringData = listOf("1", "2", null, "4", "5")
    val stringIndex = listOf("a", "b", "c", "d", "e")
    val stringSeries = Series(stringData, stringIndex)

    println(stringSeries) // Pretty-printed output
    println("Max: ${stringSeries.max()}") // Max value
    println("Min: ${stringSeries.min()}") // Min value

    val filteredStringSeries = stringSeries.filter { it != null && (it as String) > "2" }
    println(filteredStringSeries) // Filtered series

    val filledStringSeries = stringSeries.fillna("0")
    println(filledStringSeries) // Series with nulls filled
}