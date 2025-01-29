package sklearn.svm
//
//import pandas.DataFrame
//import pandas.read_csv
//import sklearn.ensemble.RandomForestClassifier
//import sklearn.model_selection.train_test_split
//import kotlin.math.max
//import kotlin.random.Random
//
//// Data class to represent a single data point
//data class DataPoint(val features: DoubleArray, val label: Int)
//
//// Linear SVM Classifier
//class SVC(private val learningRate: Double = 0.01, private val epochs: Int = 1000, private val lambda: Double = 0.01) {
//
//    private var weights: DoubleArray = doubleArrayOf()
//    private var bias: Double = 0.0
//
//    // Train the SVM
//    fun train(data: List<DataPoint>) {
//        if (data.isEmpty()) throw IllegalArgumentException("Data cannot be empty.")
//
//        val numFeatures = data[0].features.size
//        weights = DoubleArray(numFeatures) { Random.nextDouble(-1.0, 1.0) } // Initialize weights randomly
//        bias = Random.nextDouble(-1.0, 1.0) // Initialize bias randomly
//
//        for (epoch in 1..epochs) {
//            var totalLoss = 0.0
//            for (point in data) {
//                val prediction = predict(point.features)
//                val loss = hingeLoss(point.label, prediction)
//                totalLoss += loss
//
//                // Update weights and bias using gradient descent
//                if (loss > 0) {
//                    for (i in weights.indices) {
//                        weights[i] -= learningRate * (lambda * weights[i] - point.label * point.features[i])
//                    }
//                    bias -= learningRate * (-point.label)
//                } else {
//                    for (i in weights.indices) {
//                        weights[i] -= learningRate * lambda * weights[i]
//                    }
//                }
//            }
//
//            // Print loss for monitoring
//            if (epoch % 100 == 0) {
//                println("Epoch $epoch: Loss = ${totalLoss / data.size}")
//            }
//        }
//    }
//
//    // Predict the label for a single data point
//    fun predict(features: DoubleArray): Double {
//        return dotProduct(weights, features) + bias
//    }
//
//    // Classify the label (returns -1 or 1)
//    fun classify(features: DoubleArray): Int {
//        return if (predict(features) >= 0) 1 else -1
//    }
//
//    // Calculate the hinge loss
//    private fun hingeLoss(trueLabel: Int, prediction: Double): Double {
//        return max(0.0, 1 - trueLabel * prediction)
//    }
//
//    // Helper function to compute the dot product of two vectors
//    private fun dotProduct(a: DoubleArray, b: DoubleArray): Double {
//        return a.zip(b).sumOf { (x, y) -> x * y }
//    }
//}
//
//// Main function
//fun main() {
//    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
//    val df = read_csv(filePath)
//
//    val targetColumn = "target"
//
//    val dataSet = train_test_split(
//        df.drop(targetColumn),
//        df[targetColumn],
//        test_size = 0.2,
//        random_state = 42
//    )
//
//    // Step 4: Train the SVM model
////    val svc = SVC(learningRate = 0.01, epochs = 1000, lambda = 0.01)
////    svc.train(trainData)
////
////    var correctPredictions = 0
////    for ((test_features, test_label) in dataSet.X_test.zip(dataSet.y_test)) {
////        val prediction = svc.predict(test_features)
////        if (prediction == test_label) {
////            correctPredictions++
////        }
////    }
////    val accuracy = correctPredictions.toDouble() / dataSet.y_test.size
////    println("Accuracy: $accuracy")
//}