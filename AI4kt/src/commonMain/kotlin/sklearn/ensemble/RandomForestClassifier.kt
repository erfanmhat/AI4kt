package io.ai4kt.ai4kt.fibonacci.sklearn.ensemble

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.metrics.accuracy_score
import io.ai4kt.ai4kt.fibonacci.sklearn.model_selection.train_test_split
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.distinct
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
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

    fun fit(X: D2Array<Double>, y: D1Array<Int>) {
        root = buildTree(X = X, y = y, depth = 0)
    }

    // Recursively build the tree
    private fun buildTree(X: D2Array<Double>, y: D1Array<Int>, depth: Int): TreeNode {
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
            if (X[i, bestFeatureIndex] < bestThreshold) {
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
                X = X.selectRowD2Array(leftIndices),
                y = y.selectRowD1Array(leftIndices),
                depth = depth + 1
            ),
            right = buildTree(
                X = X.selectRowD2Array(rightIndices),
                y = y.selectRowD1Array(rightIndices),
                depth = depth + 1
            )
        )
    }

    // Find the best split for a node
    private fun findBestSplit(X: D2Array<Double>, y: D1Array<Int>, numFeatures: Int): Pair<Int, Double> {
        var bestFeatureIndex = -1
        var bestThreshold = 0.0
        var bestImpurity = Double.MAX_VALUE

        // Try random features
        val featureIndices = (0 until numFeatures).shuffled().take(sqrt(numFeatures.toDouble()).toInt())
        for (featureIndex in featureIndices) {
            val thresholds = X[0..<X.shape[0], featureIndex].distinct()
            for (threshold in thresholds) {
                val leftIndices = mutableListOf<Int>()
                val rightIndices = mutableListOf<Int>()
                for (i in 0 until y.shape[0]) {
                    if (X[i, featureIndex] < threshold) {
                        leftIndices.add(i)
                    } else {
                        rightIndices.add(i)
                    }
                }
                val impurity = weightedImpurity(
                    y_left = y.selectRowD1Array(leftIndices),
                    y_right = y.selectRowD1Array(rightIndices),
                )
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
    private fun weightedImpurity(y_left: D1Array<Int>, y_right: D1Array<Int>): Double {
        val n = y_left.shape[0] + y_right.shape[0]
        val pLeft = y_left.shape[0].toDouble() / n
        val pRight = y_right.shape[0].toDouble() / n
        return pLeft * giniImpurity(y_left) + pRight * giniImpurity(y_right)
    }

    // Calculate Gini impurity
    private fun giniImpurity(y: D1Array<Int>): Double {
        val labelCounts = y.toList().groupingBy { it }.eachCount()
        val total = y.shape[0].toDouble()
        return 1.0 - labelCounts.values.sumOf { (it / total).pow(2) }
    }

    // Get the majority label in a dataset
    private fun majorityLabel(y: D1Array<Int>): Int {
        return y.toList().groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0
    }

    // Predict the label for a single data point
    fun predict(features: D1Array<Double>): Int {
        return predictNode(root, features)
    }

    // Recursively predict using the tree
    private fun predictNode(node: TreeNode?, features: D1Array<Double>): Int {
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
    fun fit(X: D2Array<Double>, y: D1Array<Int>) {
        for (i in 0 until numTrees) {
            val bootstrapSampleIndices = List(y.shape[0]) { (0 until y.shape[0]).random() }
            val bootstrapSamples_X = mk.zeros<Double>(bootstrapSampleIndices.size, X.shape[1])
            for ((i, idx) in bootstrapSampleIndices.withIndex()) {
                bootstrapSamples_X[i] = X[idx]
            }
            val bootstrapSamples_y = mk.zeros<Int>(bootstrapSampleIndices.size)
            for ((i, idx) in bootstrapSampleIndices.withIndex()) {
                bootstrapSamples_y[i] = y[idx]
            }
            val tree = DecisionTree(maxDepth, minSamplesSplit)
            tree.fit(X = bootstrapSamples_X, y = bootstrapSamples_y)
            trees.add(tree)
        }
    }

    // Predict the label for a single data point
    fun predict(X: D2Array<Double>): D1Array<Int> {
        val result = mutableListOf<Int>()
        for (index in 0 until X.shape[0]) {
            val predictions = trees.map { it.predict(X[index] as D1Array<Double>) }
            result.add(predictions.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0)
        }
        return mk.ndarray(result)
    }
}

// Main function
fun main() {
    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
    val df = read_csv(filePath)

    val targetColumn = "target"

    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Int>,
        test_size = 0.2,
        random_state = 42
    )

    val randomForest = RandomForestClassifier(numTrees = 100, maxDepth = 10, minSamplesSplit = 2)
    randomForest.fit(X = dataSet.X_train, y = dataSet.y_train)

    val y_pred = randomForest.predict(dataSet.X_test)
    val accuracy = accuracy_score(dataSet.y_test, y_pred)
    println("Accuracy: $accuracy")
}

fun D2Array<Double>.selectRowD2Array(indices: MutableList<Int>): D2Array<Double> {
    return mk.ndarray(indices.map { this[it, 0..<this.shape[1]].toList() }.toList())
}

fun D1Array<Int>.selectRowD1Array(indices: MutableList<Int>): D1Array<Int> {
    return mk.ndarray(indices.map { this[it] }.toList())
}