package io.ai4kt.ai4kt.sklearn.svm

import io.ai4kt.ai4kt.pandas.DataFrame
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import kotlin.math.max
import kotlin.random.Random

// Data class to represent a single data point
data class DataPoint(val features: DoubleArray, val label: Int)

// Linear SVM Classifier
class SVC(private val learningRate: Double = 0.01, private val epochs: Int = 1000, private val lambda: Double = 0.01) {

    private var weights: DoubleArray = doubleArrayOf()
    private var bias: Double = 0.0

    // Train the SVM
    fun train(data: List<DataPoint>) {
        if (data.isEmpty()) throw IllegalArgumentException("Data cannot be empty.")

        val numFeatures = data[0].features.size
        weights = DoubleArray(numFeatures) { Random.nextDouble(-1.0, 1.0) } // Initialize weights randomly
        bias = Random.nextDouble(-1.0, 1.0) // Initialize bias randomly

        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            for (point in data) {
                val prediction = predict(point.features)
                val loss = hingeLoss(point.label, prediction)
                totalLoss += loss

                // Update weights and bias using gradient descent
                if (loss > 0) {
                    for (i in weights.indices) {
                        weights[i] -= learningRate * (lambda * weights[i] - point.label * point.features[i])
                    }
                    bias -= learningRate * (-point.label)
                } else {
                    for (i in weights.indices) {
                        weights[i] -= learningRate * lambda * weights[i]
                    }
                }
            }

            // Print loss for monitoring
            if (epoch % 100 == 0) {
                println("Epoch $epoch: Loss = ${totalLoss / data.size}")
            }
        }
    }

    // Predict the label for a single data point
    fun predict(features: DoubleArray): Double {
        return dotProduct(weights, features) + bias
    }

    // Classify the label (returns -1 or 1)
    fun classify(features: DoubleArray): Int {
        return if (predict(features) >= 0) 1 else -1
    }

    // Calculate the hinge loss
    private fun hingeLoss(trueLabel: Int, prediction: Double): Double {
        return max(0.0, 1 - trueLabel * prediction)
    }

    // Helper function to compute the dot product of two vectors
    private fun dotProduct(a: DoubleArray, b: DoubleArray): Double {
        return a.zip(b).sumOf { (x, y) -> x * y }
    }
}

// Helper function to convert DataFrame to List<DataPoint>
fun DataFrame.toDataPoints(targetColumn: String): List<DataPoint> {
    val featureColumns = columns.filter { it != targetColumn }
    return (0 until rowCount).map { i ->
        val features = featureColumns.map { getColumn(it)[i]?.toString()?.toDouble() ?: 0.0 }.toDoubleArray()
        val label = getColumn(targetColumn)[i] as Int
        DataPoint(features, label)
    }
}

// Main function
fun main() {
    // Step 1: Load the dataset into a DataFrame
    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
    val df = read_csv(filePath)

    // Step 2: Convert DataFrame to List<DataPoint>
    val targetColumn = "target"
    val data = df.toDataPoints(targetColumn)

    // Step 3: Split the dataset into training and testing sets (80% train, 20% test)
    val splitIndex = (data.size * 0.8).toInt()
    val trainData = data.subList(0, splitIndex)
    val testData = data.subList(splitIndex, data.size)

    // Step 4: Train the SVM model
    val svc = SVC(learningRate = 0.01, epochs = 1000, lambda = 0.01)
    svc.train(trainData)

    // Step 5: Evaluate the model
    var correctPredictions = 0
    for (dataPoint in testData) {
        val prediction = svc.classify(dataPoint.features)
        if (prediction == dataPoint.label) {
            correctPredictions++
        }
    }
    val accuracy = correctPredictions.toDouble() / testData.size
    println("Accuracy: $accuracy")
}