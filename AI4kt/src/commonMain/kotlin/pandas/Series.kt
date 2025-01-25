package io.ai4kt.ai4kt.fibonacci.pandas

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.reflect.KClass

class Series<T>(private val data: List<T?>, private val index: List<Any>? = null) : Iterable<T?> {
    init {
        if (index != null && index.size != data.size) {
            throw IllegalArgumentException("Index length must match data length.")
        }
    }

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

    fun <T : Any> getValues(clazz: KClass<T>): D1Array<T> {
        return when (clazz) {
            Double::class -> mk.ndarray(data as List<Double>) as D1Array<T>
            Int::class -> mk.ndarray(data as List<Int>) as D1Array<T>
            else -> throw IllegalArgumentException("Unsupported type: ${clazz::class}")
        }
    }

    fun filter(condition: (Any?) -> Boolean): Series<T> {
        val filteredData = data.filter(condition)
        val filteredIndex = index?.let { idx ->
            data.indices.filter { condition(data[it]) }.map { idx[it] }
        }
        return Series(filteredData, filteredIndex)
    }

    operator fun <T> plus(other: Series<T>): Series<Double> {
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
        return data.filterNotNull() // Filter out null values
            .filterIsInstance<Number>() // Filter only numeric types
            .sumOf { it.toDouble() } // Sum all numeric values as Double
    }

    fun mean(): Double {
        if (data.isEmpty()) throw NoSuchElementException("List is empty")
        return sum() / data.filterNotNull().filterIsInstance<Number>().size
    }

    fun min(): Double {
        if (data.isEmpty()) throw NoSuchElementException("List is empty")
        return data.filterNotNull() // Filter out null values
            .filterIsInstance<Number>() // Filter only numeric types
            .minOf { it.toDouble() } // Find the minimum value
    }

    fun max(): Double {
        if (data.isEmpty()) throw NoSuchElementException("List is empty")
        return data.filterNotNull() // Filter out null values
            .filterIsInstance<Number>() // Filter only numeric types
            .maxOf { it.toDouble() } // Find the maximum value
    }

    fun std(): Double {
        val mean = mean()
        val variance = data.filterNotNull()
            .filterIsInstance<Number>()
            .map { (it.toDouble() - mean).pow(2) }
            .average()
        return sqrt(variance)
    }

    fun get_var(): Double {
        val mean = mean()
        return data.filterNotNull()
            .filterIsInstance<Number>()
            .map { (it.toDouble() - mean).pow(2) }
            .average()
    }

    fun dropna(): Series<T> {
        val nonNullData = data.filterNotNull()
        val nonNullIndex = index?.let { idx ->
            data.indices.filter { data[it] != null }.map { idx[it] }
        }
        return Series(nonNullData, nonNullIndex)
    }

    fun fillna(value: T): Series<T> {
        val filledData = data.map { it ?: value }
        return Series(filledData, index)
    }

    fun toList(): List<Any?> {
        return data.toList()
    }

    fun toMap(): Map<Any, Any?> {
        if (index == null) throw UnsupportedOperationException("Index is not defined.")
        return index.zip(data).toMap()
    }

    override fun iterator(): Iterator<T?> {
        return data.iterator()
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
}