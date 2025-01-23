package io.ai4kt.ai4kt.fibonacci.tensorflow

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.max

class OneHotEncoding {
    fun transform(labels: D1Array<Int>, numClasses: Int? = null): D2Array<Double> {
        // Determine the number of classes
        val nClasses = numClasses ?: (labels.max()?.plus(1) ?: 0)

        // Initialize a zero matrix of shape (samples, numClasses)
        val oneHot = mk.zeros<Double>(labels.size, nClasses)

        // Fill the one-hot encoded matrix
        for ((index, label) in labels.withIndex()) {
            oneHot[index, label] = 1.0
        }

        return oneHot
    }

    fun inverseTransform(oneHot: D2Array<Double>): D1Array<Int> {
        // Convert one-hot encoded matrix back to labels
        return oneHot.mapD2ToD1Array { row -> row.indexOf(1.0) }
    }
}

fun main() {
    // Create an instance of OneHotEncoding
    val oneHotEncoder = OneHotEncoding()

    // Example labels
    val labels = mk.ndarray(mk[0, 2, 1, 2, 0])

    // Transform labels to one-hot encoded format
    val oneHotEncoded = oneHotEncoder.transform(labels)
    println("One-Hot Encoded:")
    println(oneHotEncoded)

    // Inverse transform one-hot encoded matrix back to labels
    val originalLabels = oneHotEncoder.inverseTransform(oneHotEncoded)
    println("Original Labels:")
    println(originalLabels)
}