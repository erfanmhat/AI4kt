package io.ai4kt.ai4kt.fibonacci.sklearn.ensemble

import io.ai4kt.ai4kt.fibonacci.numpy.ndarray
import io.ai4kt.ai4kt.fibonacci.numpy.to_ndarray
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.model_selection.train_test_split
import kotlin.math.pow
import kotlin.math.sqrt

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

    fun fit(X: ndarray<Double>, y: ndarray<Int>) {
        root = buildTree(X = X, y = y, depth = 0)
    }

    // Recursively build the tree
    private fun buildTree(X: ndarray<Double>, y: ndarray<Int>, depth: Int): TreeNode {
        val numSamples = y.shape[0]
        val numFeatures = X.shape[1]

        // Stop if max depth is reached or too few samples
        if (depth >= maxDepth || numSamples < minSamplesSplit) {
            return TreeNode(label = majorityLabel(y))
        }

        // Find the best split
        val (bestFeatureIndex, bestThreshold) = findBestSplit(X, y, numFeatures)

        // Split the data
        val leftIndices = mutableListOf<Int>()
        val rightIndices = mutableListOf<Int>()
        for (i in 0 until numSamples) {
            if ((X[i, bestFeatureIndex].data[0] as Double) < bestThreshold) {
                leftIndices.add(i)
            } else {
                rightIndices.add(i)
            }
        }

        // Stop if no split improves the impurity
        if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
            return TreeNode(label = majorityLabel(y))
        }

        // Recursively build the left and right subtrees
        return TreeNode(
            featureIndex = bestFeatureIndex,
            threshold = bestThreshold,
            left = buildTree(
                X = X[leftIndices].to_ndarray(),
                y = y[leftIndices].to_ndarray(),
                depth = depth + 1
            ),
            right = buildTree(
                X = X[rightIndices].to_ndarray(),
                y = y[rightIndices].to_ndarray(),
                depth = depth + 1
            )
        )
    }

    // Find the best split for a node
    private fun findBestSplit(X: ndarray<Double>, y: ndarray<Int>, numFeatures: Int): Pair<Int, Double> {
        var bestFeatureIndex = -1
        var bestThreshold = 0.0
        var bestImpurity = Double.MAX_VALUE

        // Try random features
        val featureIndices = (0 until numFeatures).shuffled().take(sqrt(numFeatures.toDouble()).toInt())
        for (featureIndex in featureIndices) {
            val thresholds = X[null, featureIndex].distinct()
            for (threshold in thresholds) {
                val leftIndices = mutableListOf<Int>()
                val rightIndices = mutableListOf<Int>()
                for (i in 0 until y.shape[0]) {
                    if ((X[i, featureIndex].data[0] as Double) < threshold.data[0] as Double) {
                        leftIndices.add(i)
                    } else {
                        rightIndices.add(i)
                    }
                }
                val impurity = weightedImpurity(
                    y_left = y[leftIndices].to_ndarray(),
                    y_right = y[rightIndices].to_ndarray()
                )
                if (impurity < bestImpurity) {
                    bestImpurity = impurity
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold.data[0] as Double
                }
            }
        }

        return Pair(bestFeatureIndex, bestThreshold)
    }

    // Calculate weighted impurity (Gini index)
    private fun weightedImpurity(y_left: ndarray<Int>, y_right: ndarray<Int>): Double {
        val n = y_left.shape[0] + y_right.shape[0]
        val pLeft = y_left.shape[0].toDouble() / n
        val pRight = y_right.shape[0].toDouble() / n
        return pLeft * giniImpurity(y_left) + pRight * giniImpurity(y_right)
    }

    // Calculate Gini impurity
    private fun giniImpurity(y: ndarray<Int>): Double {
        val labelCounts = y.groupingBy { it }.eachCount()
        val total = y.shape[0].toDouble()
        return 1.0 - labelCounts.values.sumOf { (it / total).pow(2) }
    }

    // Get the majority label in a dataset
    private fun majorityLabel(y: ndarray<Int>): Int {
        return y.groupingBy { it.data[0] as Int }.eachCount().maxByOrNull { it.value }?.key ?: 0
    }

    // Predict the label for a single data point
    fun predict(features: ndarray<Double>): Int {
        return predictNode(root, features)
    }

    // Recursively predict using the tree
    private fun predictNode(node: TreeNode?, features: ndarray<Double>): Int {
        if (node?.label != null) {
            return node.label
        }
        return if ((features[node!!.featureIndex!!].data[0] as Double) < node.threshold!!) {
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
    fun fit(X: ndarray<Double>, y: ndarray<Int>) {
        for (i in 0 until numTrees) {
            val bootstrapSampleIndices = List(y.shape[0]) { (0 until y.shape[0]).random() }
            val tree = DecisionTree(maxDepth, minSamplesSplit)
            tree.fit(
                X = X[bootstrapSampleIndices].to_ndarray(),
                y = y[bootstrapSampleIndices].to_ndarray()
            )
            trees.add(tree)
        }
    }

    // Predict the label for a single data point
    fun predict(features: ndarray<Double>): Int {
        val predictions = trees.map { it.predict(features) }
        return predictions.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0
    }
}

// Main function
fun main() {
    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
    val df = read_csv(filePath)

    val targetColumn = "target"

    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn],
        test_size = 0.2,
        random_state = 42
    )

    val randomForest = RandomForestClassifier(numTrees = 100, maxDepth = 10, minSamplesSplit = 2)
    randomForest.fit(X = dataSet.X_train, y = dataSet.y_train)

    var correctPredictions = 0
    for (i in 0 until dataSet.X_test.shape[0]) {
        val prediction = randomForest.predict(dataSet.X_test[i])
        if (prediction == dataSet.y_test[i].data[0] as Int) {
            correctPredictions++
        }
    }
    val accuracy = correctPredictions.toDouble() / dataSet.y_test.shape[0]
    println("Accuracy: $accuracy")
}