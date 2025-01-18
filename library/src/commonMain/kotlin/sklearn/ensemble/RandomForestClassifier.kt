package io.ai4kt.ai4kt.fibonacci.sklearn.ensemble

import io.ai4kt.ai4kt.pandas.DataFrame
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import kotlin.math.pow
import kotlin.math.sqrt

// Data class to represent a single data point
data class DataPoint(val features: DoubleArray, val label: Int)

// Decision Tree Node
class TreeNode(
    val featureIndex: Int? = null, // Feature index to split on
    val threshold: Double? = null, // Threshold for the split
    val left: TreeNode? = null, // Left subtree
    val right: TreeNode? = null, // Right subtree
    val label: Int? = null // Label if this is a leaf node
)

// Decision Tree
class DecisionTree(private val maxDepth: Int = 10, private val minSamplesSplit: Int = 2) {

    private var root: TreeNode? = null

    // Train the decision tree
    fun train(data: List<DataPoint>) {
        root = buildTree(data, depth = 0)
    }

    // Recursively build the tree
    private fun buildTree(data: List<DataPoint>, depth: Int): TreeNode? {
        val numSamples = data.size
        val numFeatures = data[0].features.size

        // Stop if max depth is reached or too few samples
        if (depth >= maxDepth || numSamples < minSamplesSplit) {
            return TreeNode(label = majorityLabel(data))
        }

        // Find the best split
        val (bestFeatureIndex, bestThreshold) = findBestSplit(data, numFeatures)

        // Split the data
        val (leftData, rightData) = data.partition { it.features[bestFeatureIndex] < bestThreshold }

        // Stop if no split improves the impurity
        if (leftData.isEmpty() || rightData.isEmpty()) {
            return TreeNode(label = majorityLabel(data))
        }

        // Recursively build the left and right subtrees
        return TreeNode(
            featureIndex = bestFeatureIndex,
            threshold = bestThreshold,
            left = buildTree(leftData, depth + 1),
            right = buildTree(rightData, depth + 1)
        )
    }

    // Find the best split for a node
    private fun findBestSplit(data: List<DataPoint>, numFeatures: Int): Pair<Int, Double> {
        var bestFeatureIndex = -1
        var bestThreshold = 0.0
        var bestImpurity = Double.MAX_VALUE

        // Try random features
        val featureIndices = (0 until numFeatures).shuffled().take(sqrt(numFeatures.toDouble()).toInt())
        for (featureIndex in featureIndices) {
            val thresholds = data.map { it.features[featureIndex] }.distinct()
            for (threshold in thresholds) {
                val (left, right) = data.partition { it.features[featureIndex] < threshold }
                val impurity = weightedImpurity(left, right)
                if (impurity < bestImpurity) {
                    bestImpurity = impurity
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold
                }
            }
        }

        return Pair(bestFeatureIndex, bestThreshold)
    }

    // Calculate weighted impurity (Gini index)
    private fun weightedImpurity(left: List<DataPoint>, right: List<DataPoint>): Double {
        val n = left.size + right.size
        val pLeft = left.size.toDouble() / n
        val pRight = right.size.toDouble() / n
        return pLeft * giniImpurity(left) + pRight * giniImpurity(right)
    }

    // Calculate Gini impurity
    private fun giniImpurity(data: List<DataPoint>): Double {
        val labelCounts = data.groupingBy { it.label }.eachCount()
        val total = data.size.toDouble()
        return 1.0 - labelCounts.values.sumOf { (it / total).pow(2) }
    }

    // Get the majority label in a dataset
    private fun majorityLabel(data: List<DataPoint>): Int {
        return data.groupingBy { it.label }.eachCount().maxByOrNull { it.value }?.key ?: 0
    }

    // Predict the label for a single data point
    fun predict(features: DoubleArray): Int {
        return predictNode(root, features)
    }

    // Recursively predict using the tree
    private fun predictNode(node: TreeNode?, features: DoubleArray): Int {
        if (node?.label != null) {
            return node.label
        }
        return if (features[node!!.featureIndex!!] < node.threshold!!) {
            predictNode(node.left, features)
        } else {
            predictNode(node.right, features)
        }
    }
}

// Random Forest
class RandomForestClassifier(
    private val numTrees: Int = 100,
    private val maxDepth: Int = 10,
    private val minSamplesSplit: Int = 2
) {

    private val trees = mutableListOf<DecisionTree>()

    // Train the Random Forest
    fun train(data: List<DataPoint>) {
        for (i in 0 until numTrees) {
            val bootstrapSample = data.shuffled().take(data.size) // Bootstrap sampling
            val tree = DecisionTree(maxDepth, minSamplesSplit)
            tree.train(bootstrapSample)
            trees.add(tree)
        }
    }

    // Predict the label for a single data point
    fun predict(features: DoubleArray): Int {
        val predictions = trees.map { it.predict(features) }
        return predictions.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0
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

    // Step 4: Train the Random Forest model
    val randomForest = RandomForestClassifier(numTrees = 100, maxDepth = 10, minSamplesSplit = 2)
    randomForest.train(trainData)

    // Step 5: Evaluate the model
    var correctPredictions = 0
    for (dataPoint in testData) {
        val prediction = randomForest.predict(dataPoint.features)
        if (prediction == dataPoint.label) {
            correctPredictions++
        }
    }
    val accuracy = correctPredictions.toDouble() / testData.size
    println("Accuracy: $accuracy")
}