package io.ai4kt.ai4kt.fibonacci.tensorflow

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.pandas.asSeries
import io.ai4kt.ai4kt.pandas.DataFrame
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.math.max

class OneHotEncoding {
    fun transform(labels: D1Array<Int>, numClasses: Int? = null): D2Array<Double> {
        val nClasses = numClasses ?: (labels.max()?.plus(1) ?: 0)

        val oneHot = mk.zeros<Double>(labels.size, nClasses)

        for ((index, label) in labels.withIndex()) {
            oneHot[index, label] = 1.0
        }

        return oneHot
    }

    fun inverseTransform(oneHot: D2Array<Double>): D1Array<Int> {
        return oneHot.map2DRowsToIntArray { row -> row.indexOf(1.0) }
    }
}

class OneHotEncodingSeries {

    fun transform(labels: Series<Any?>, numClasses: Int? = null): D2Array<Double> {
        val uniqueLabels = labels.distinct().filterNotNull()
        val labelToIndex = uniqueLabels.withIndex().associate { it.value to it.index }
        val indexedLabels = labels.map { label -> labelToIndex[label] ?: -1 }.toIntArray()

        val nClasses = numClasses ?: max(uniqueLabels.size, indexedLabels.maxOrNull()?.plus(1) ?: 0)

        val oneHot = mk.zeros<Double>(indexedLabels.size, nClasses)

        for ((index, label) in indexedLabels.withIndex()) {
            if (label != -1) {
                oneHot[index, label] = 1.0
            }
        }

        return oneHot
    }

    fun inverseTransform(oneHot: D2Array<Double>, uniqueLabels: List<String>): List<String> {
        val labels = oneHot.map2DRowsToIntArray { row -> row.indexOf(1.0) }
        try {
            return labels.toList().map { index -> uniqueLabels.getOrNull(index)!! }
        } catch (e: NullPointerException) {
            throw Exception("One hot not found for some labels(all values of a row is zero)")
        }
    }
}

fun main() {
    val labels = listOf("cat", "dog", "cat", "bird", "dog", null).asSeries()
    val oneHotEncoder = OneHotEncodingSeries()

    val oneHot = oneHotEncoder.transform(labels)
    println("One-Hot Encoded Matrix:")
    println(oneHot)

    val uniqueLabels = labels.distinct().filterNotNull()
    val originalLabels = oneHotEncoder.inverseTransform(oneHot, uniqueLabels as List<String>)
    println("Original Labels:")
    println(originalLabels)
}

//fun main() {
//    // Create an instance of OneHotEncoding
//    val oneHotEncoder = OneHotEncoding()
//
//    // Example labels
//    val labels = mk.ndarray(mk[0, 2, 1, 2, 0])
//
//    // Transform labels to one-hot encoded format
//    val oneHotEncoded = oneHotEncoder.transform(labels)
//    println("One-Hot Encoded:")
//    println(oneHotEncoded)
//
//    // Inverse transform one-hot encoded matrix back to labels
//    val originalLabels = oneHotEncoder.inverseTransform(oneHotEncoded)
//    println("Original Labels:")
//    println(originalLabels)
//}