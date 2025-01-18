package io.ai4kt.ai4kt.pandas


import io.ai4kt.ai4kt.fibonacci.pandas.Series
import kotlin.math.pow
import kotlin.math.sqrt

class DataFrame(private val data: Map<String, Series>) {

    // Get column names
    val columns: List<String>
        get() = data.keys.toList()

    // Get number of rows
    val rowCount: Int
        get() = if (data.isNotEmpty()) data.values.first().size else 0

    // Get number of columns
    val columnCount: Int
        get() = data.size

    // Get a specific column as a Series
    fun getColumn(columnName: String): Series {
        return data[columnName] ?: throw NoSuchElementException("Column '$columnName' not found.")
    }

    // Get a specific row as a Map
    fun getRow(index: Int): Map<String, Any?> {
        if (index !in 0 until rowCount) throw IndexOutOfBoundsException("Row index out of bounds.")
        return data.mapValues { it.value[index] }
    }

    // Filter rows based on a condition
    fun filter(condition: (Map<String, Any?>) -> Boolean): DataFrame {
        val filteredIndices = (0 until rowCount).filter { index ->
            condition(getRow(index))
        }
        val filteredData = data.mapValues { (_, series) ->
            Series(filteredIndices.map { series[it] })
        }
        return DataFrame(filteredData)
    }

    // Select specific columns
    fun select(vararg columnNames: String): DataFrame {
        val selectedData =
            columnNames.associateWith { data[it] ?: throw NoSuchElementException("Column '$it' not found.") }
        return DataFrame(selectedData)
    }

    // Add a new column
    fun addColumn(columnName: String, series: Series) {
        if (series.size != rowCount) throw IllegalArgumentException("Series size must match row count.")
        data.toMutableMap()[columnName] = series
    }

    // Drop a column
    fun dropColumn(columnName: String): DataFrame {
        val newData = data.filterKeys { it != columnName }
        return DataFrame(newData)
    }

    // Sort by a column
    fun sortBy(columnName: String, ascending: Boolean = true): DataFrame {
        val series = getColumn(columnName)
        val sortedIndices = series.toList().withIndex()
            .sortedBy { (_, value) -> value as? Comparable<Any> }
            .map { it.index }
            .let { if (!ascending) it.reversed() else it }
        val sortedData = data.mapValues { (_, series) ->
            Series(sortedIndices.map { series[it] })
        }
        return DataFrame(sortedData)
    }

    // Aggregate functions
    fun sum(columnName: String): Double {
        val column = getColumn(columnName)
        return column.toList().filterIsInstance<Number>().sumOf { it.toDouble() }
    }

    fun mean(columnName: String): Double {
        val column = getColumn(columnName)
        val numbers = column.toList().filterIsInstance<Number>()
        return numbers.sumOf { it.toDouble() } / numbers.size
    }

    fun min(columnName: String): Any? {
        val column = getColumn(columnName)
        return column.toList().filterNotNull().filterIsInstance<Comparable<Any>>().minOrNull()
    }

    fun max(columnName: String): Any? {
        val column = getColumn(columnName)
        return column.toList().filterNotNull().filterIsInstance<Comparable<Any>>().maxOrNull()
    }

    fun std(columnName: String): Double {
        val mean = mean(columnName)
        val column = getColumn(columnName)
        val variance = column.toList().filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
        return sqrt(variance)
    }

    fun get_var(columnName: String): Double {
        val mean = mean(columnName)
        val column = getColumn(columnName)
        return column.toList().filterIsInstance<Number>().map { (it.toDouble() - mean).pow(2) }.average()
    }

    // Pretty-printing
    override fun toString(): String {
        val builder = StringBuilder()
        builder.append("DataFrame:\n")
        builder.append(columns.joinToString("\t") + "\n")
        for (i in 0 until rowCount) {
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

    val df = DataFrame(data)

    println(df) // Pretty-printed DataFrame

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

    val selectedDf = df.select("name", "salary")
    println("Selected columns (name, salary):")
    println(selectedDf)

    df.addColumn("bonus", Series(listOf(1000.0, 2000.0, 3000.0)))
    println("DataFrame with bonus column:")
    println(df)
}