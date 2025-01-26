package io.ai4kt.ai4kt.fibonacci.pandas

import kotlinx.io.files.SystemFileSystem
import io.ai4kt.ai4kt.pandas.DataFrame
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.readString

fun read_csv(filePath: String, delimiter: String = ","): DataFrame {
    val file = SystemFileSystem.source(Path(filePath)).buffered()
    val lines = file.readString().trim().split("\n")
    if (lines.isEmpty()) throw IllegalArgumentException("File is empty: $filePath")

    // Extract header (column names)
    val header = lines[0].split(delimiter).map { it.trim() }

    // Extract data rows
    val data = lines.subList(1, lines.size).map { line ->
        val values = line.split(delimiter).map { it.trim() }
        header.zip(values).toMap()
    }

    // Convert data to a map of columns with automatic type detection
    val columnData: Map<String, Series<Any?>> = header.associateWith { columnName ->
        val columnValues = data.map { row -> row[columnName] }
        val convertedValues = when {
            columnValues.all { it?.isInt() == true } -> columnValues.map { it?.toIntOrNull() } // Convert to Int
            columnValues.all { it?.isNumeric() == true } -> columnValues.map { it?.toDoubleOrNull() } // Convert to Double
            else -> columnValues // Keep as String if not all values are numeric
        }
        Series(convertedValues)
    }

    return DataFrame(columnData.toMutableMap())
}

// Helper function to check if a string is an integer
fun String?.isInt(): Boolean {
    return this?.toIntOrNull() != null
}

// Helper function to check if a string is numeric
fun String?.isNumeric(): Boolean {
    return this?.toDoubleOrNull() != null
}

fun main() {
    val df = read_csv("D:\\repo\\AI4kt\\data\\test_data.csv")

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

    val selectedDf = df[listOf("name", "salary")]
    println("Selected columns (name, salary):")
    println(selectedDf)

    df["bonus"] = Series(listOf(1000.0, 2000.0, 3000.0))
    println("DataFrame with bonus column:")
    println(df)
}