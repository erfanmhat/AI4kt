package io.ai4kt.ai4kt.pandas


import io.ai4kt.ai4kt.fibonacci.numpy.ndarray
import io.ai4kt.ai4kt.fibonacci.pandas.Series
import kotlin.math.pow
import kotlin.math.sqrt

class DataFrame(private val data: MutableMap<String, Series>) {

    val columns: List<String>
        get() = data.keys.toList()

    val shape: List<Int>
        get() = listOf(if (data.isNotEmpty()) data.values.first().shape[0] else 0, data.size)

    val values: ndarray<Any>
        get() {
            if (data.isEmpty()) {
                return ndarray(shape = listOf(0, 0), initialValue = null as Any?)
            }

            val numRows = data.values.first().shape[0]
            val numCols = data.size

            val result = ndarray<Any>(shape = listOf(numRows, numCols), initialValue = null as Any?)

            for ((colIndex, columnName) in columns.withIndex()) {
                val series = data[columnName]!!
                for (rowIndex in 0 until numRows) {
                    result[rowIndex, colIndex] = series[rowIndex]!!
                }
            }

            return result
        }

    fun iloc(index: Int): Map<String, Any?> {
        if (index !in 0 until shape[1]) throw IndexOutOfBoundsException("Row index out of bounds.")
        return data.mapValues { it.value[index] }
    }

    fun filter(condition: (Map<String, Any?>) -> Boolean): DataFrame {
        val filteredIndices = (0 until shape[1]).filter { index ->
            condition(iloc(index))
        }
        val filteredData = data.mapValues { (_, series) ->
            Series(filteredIndices.map { series[it] })
        }
        return DataFrame(filteredData.toMutableMap())
    }

    operator fun get(columnName: String): Series {
        return data[columnName] ?: throw NoSuchElementException("Column '$columnName' not found.")
    }

    operator fun set(key: String, value: Series) {
        if (value.shape[0] != shape[1]) throw IllegalArgumentException("Series size must match row count.")
        data[key] = value
    }

    operator fun get(columnNames: List<String>): DataFrame {
        val selectedData =
            columnNames.associateWith { data[it] ?: throw NoSuchElementException("Column '$it' not found.") }
        return DataFrame(selectedData.toMutableMap())
    }

    fun drop(columnName: String): DataFrame {
        val newData = data.filterKeys { it != columnName }
        return DataFrame(newData.toMutableMap())
    }

    fun drop(columnNames: List<String>): DataFrame {
        val newData = data.filterKeys { it !in columnNames }
        return DataFrame(newData.toMutableMap())
    }

    fun sortBy(columnName: String, ascending: Boolean = true): DataFrame {
        val series = this[columnName]
        val sortedIndices = series.toList().withIndex()
            .sortedBy { (_, value) -> value as? Comparable<Any> }
            .map { it.index }
            .let { if (!ascending) it.reversed() else it }
        val sortedData = data.mapValues { (_, series) ->
            Series(sortedIndices.map { series[it] })
        }
        return DataFrame(sortedData.toMutableMap())
    }

    fun sum(columnName: String): Double {
        val column = this[columnName]
        return column.toList().filterIsInstance<Number>().sumOf { it.toDouble() }
    }

    fun mean(columnName: String): Double {
        val column = this[columnName]
        val numbers = column.toList().filterIsInstance<Number>()
        return numbers.sumOf { it.toDouble() } / numbers.size
    }

    fun min(columnName: String): Any? {
        val column = this[columnName]
        return column.toList().filterNotNull().filterIsInstance<Comparable<Any>>().minOrNull()
    }

    fun max(columnName: String): Any? {
        val column = this[columnName]
        return column.toList().filterNotNull().filterIsInstance<Comparable<Any>>().maxOrNull()
    }

    fun std(columnName: String): Double {
        val mean = mean(columnName)
        val column = this[columnName]
        val variance = column.toList().filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
        return sqrt(variance)
    }

    fun get_var(columnName: String): Double {
        val mean = mean(columnName)
        val column = this[columnName]
        return column.toList().filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
    }

    /**
     * Returns a subset of the DataFrame based on the given indices.
     *
     * @param indices The indices of the rows to include in the subset.
     * @return A new DataFrame containing the specified rows.
     */
    fun slice(indices: List<Int>): DataFrame {
        val slicedData = mutableMapOf<String, Series>()
        for ((columnName, series) in this.data) {
            // از تابع slice در Series استفاده می‌کنیم
            val slicedSeries = series.slice(indices)
            slicedData[columnName] = slicedSeries
        }
        return DataFrame(slicedData)
    }

    override fun toString(): String {
        val builder = StringBuilder()
        builder.append("DataFrame:\n")
        builder.append(columns.joinToString("\t") + "\n")
        for (i in 0 until shape[1]) {
            builder.append(data.values.map { it[i] }.joinToString("\t") + "\n")
        }
        return builder.toString()
    }
}

fun main() {
    val data = mapOf(
        "name" to Series(listOf("Alice", "Bob", "Charlie")),
        "age" to Series(listOf(25, 30, 35)),
        "salary" to Series(listOf(50000.0, 60000.0, 70000.0))
    )

    val df = DataFrame(data.toMutableMap())

    println(df)

    println("Sum of salaries: ${df.sum("salary")}")
    println("Mean age: ${df.mean("age")}")
    println("Max salary: ${df.max("salary")}")
    println("Min age: ${df.min("age")}")

    val filteredDf = df.filter { (it["age"] as Int) > 25 }
    println("Filtered DataFrame (age > 25):")
    println(filteredDf)

    val sortedDf = df.sortBy("salary", ascending = false)
    println("Sorted DataFrame (salary descending):")
    println(sortedDf)

    val selectedDf = df[listOf("name", "salary")]
    println("Selected columns (name, salary):")
    println(selectedDf)

    df["bonus"] = Series(listOf(1000.0, 2000.0, 3000.0))
    println("DataFrame with bonus column:")
    println(df)
}