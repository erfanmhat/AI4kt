package io.ai4kt.ai4kt.pandas


import io.ai4kt.ai4kt.fibonacci.pandas.Series
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.reflect.KClass

class DataFrame(private val data: MutableMap<String, Series<Any?>>) {

    val columns: List<String>
        get() = data.keys.toList()

    val shape: List<Int>
        get() = listOf(if (data.isNotEmpty()) data.values.first().shape[0] else 0, data.size)

    val dtypes: List<String>
        get() = data.map { it.value.dtype }


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
        return DataFrame(filteredData.toMutableMap() as MutableMap<String, Series<Any?>>)
    }

    operator fun get(columnName: String): Series<Any?> {
        return data[columnName] ?: throw NoSuchElementException("Column '$columnName' not found.")
    }

    operator fun set(key: String, value: Series<Any?>) {
        if (value.shape[0] != shape[0]) throw IllegalArgumentException("Series size must match row count.")
        data[key] = value
    }

    operator fun get(columnNames: List<String>): DataFrame {
        val selectedData =
            columnNames.associateWith { data[it] ?: throw NoSuchElementException("Column '$it' not found.") }
        return DataFrame(selectedData.toMutableMap())
    }

    fun <T : Any> getValues(clazz: KClass<T>): D2Array<T> {
        return when (clazz) {
            Double::class -> (mk.ndarray(data.map { it.value.toList() } as List<List<Double>>) as D2Array<T>).transpose()
            Int::class -> (mk.ndarray(data.map { it.value.toList() } as List<List<Int>>) as D2Array<T>).transpose()
            else -> throw IllegalArgumentException("Unsupported type: ${clazz::class}")
        }
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
        return DataFrame(sortedData.toMutableMap() as MutableMap<String, Series<Any?>>)
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

    override fun toString(): String {
        val builder = StringBuilder()
        builder.append("DataFrame:\n")

        // Map to store the padding required for each column
        val columnMaxPaddingMap = mutableMapOf<String, Int>()

        // Calculate padding for each column header
        for (i in 0 until shape[1]) {
            builder.append(columns[i])
            columnMaxPaddingMap[columns[i]] = max(
                calculate_max_len_of_column(data[columns[i]]?.data ?: listOf()),
                columns[i].length
            ) + 1

            builder.append(" ".repeat(columnMaxPaddingMap[columns[i]]!! - columns[i].length))
        }
        builder.append("\n")

        // Append rows with proper padding
        for (i in 0 until shape[0]) {
            for (j in 0 until shape[1]) {
                val columnName = columns[j]
                val cellValue = data[columnName]!![i].toString()
                val padding = columnMaxPaddingMap[columnName]!!

                builder.append(cellValue)
                builder.append(" ".repeat(padding - cellValue.length))
            }
            builder.append("\n") // Move to the next row
        }

        return builder.toString()
    }
}

fun <T> calculate_max_len_of_column(arr: List<T>): Int {
    var result = 0
    for (index in arr.indices) {
        result = max(arr[index].toString().length, result)
    }
    return result
}

fun main() {
    val data = mapOf(
        "name" to Series(listOf("Alice", "Bob", "Charlie")),
        "age" to Series(listOf(25, 30, 35)),
        "salary" to Series(listOf(50000.0, 60000.0, 70000.0))
    )

    val df = DataFrame(data.toMutableMap() as MutableMap<String, Series<Any?>>)

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

//    val selectedDf = df[listOf("name", "salary")]
//    println("Selected columns (name, salary):")
//    println(selectedDf)

    df["bonus"] = Series(listOf(1000.0, 2000.0, 3000.0))
    println("DataFrame with bonus column:")
    println(df)
}