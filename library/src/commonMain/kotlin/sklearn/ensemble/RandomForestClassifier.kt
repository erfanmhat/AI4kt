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
    private fun buildTree(X: ndarray<Double>, y: ndarray<Int>, depth: Int): TreeNode? {
        val numSamples = y.shape[0]
        val numFeatures = X.shape[1]

        // Stop if max depth is reached or too few samples
        if (depth >= maxDepth || numSamples < minSamplesSplit) {
            return TreeNode(label = majorityLabel(X = X, y = y))
        }

        // Find the best split
        val (bestFeatureIndex, bestThreshold) = findBestSplit(X = X, y = y, numFeatures = numFeatures)

        // Split the data
        val (leftData, rightData) = X.zip(y)
            .partition { (it.first[bestFeatureIndex].data[0] as Double) < bestThreshold }

        // Stop if no split improves the impurity
        if (leftData.isEmpty() || rightData.isEmpty()) {
            return TreeNode(label = majorityLabel(X = X, y = y))
        }

        // Recursively build the left and right subtrees
        return TreeNode(
            featureIndex = bestFeatureIndex,
            threshold = bestThreshold,
            left = buildTree(
                X = leftData.map { it.first }.to_ndarray(),
                y = leftData.map { it.second }.to_ndarray(),
                depth = depth + 1
            ),
            right = buildTree(
                X = rightData.map { it.first }.to_ndarray(),
                y = rightData.map { it.second }.to_ndarray(),
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
                val (left, right) = X.zip(y)
                    .partition { (it.first[featureIndex].data[0] as Double) < threshold.data[0] as Double }
                val impurity = weightedImpurity(
                    X_left = left.map { it.first }.to_ndarray(),
                    y_left = left.map { it.second }.to_ndarray(),
                    X_right = right.map { it.first }.to_ndarray(),
                    y_right = right.map { it.second }.to_ndarray()
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
    private fun weightedImpurity(
        X_left: ndarray<Double>,
        y_left: ndarray<Int>,
        X_right: ndarray<Double>,
        y_right: ndarray<Int>
    ): Double {
        val n = X_left.shape[0] + X_right.shape[0]
        val pLeft = X_left.shape[0].toDouble() / n
        val pRight = X_right.shape[0].toDouble() / n
        return pLeft * giniImpurity(X_left, y_left) + pRight * giniImpurity(X_right, y_right)
    }

    // Calculate Gini impurity
    private fun giniImpurity(X: ndarray<Double>, y: ndarray<Int>): Double {
        val labelCounts = X.zip(y).groupingBy { it.second }.eachCount()
        val total = X.shape[0].toDouble()
        return 1.0 - labelCounts.values.sumOf { (it / total).pow(2) }
    }

    // Get the majority label in a dataset
    private fun majorityLabel(X: ndarray<Double>, y: ndarray<Int>): Int {
        return X.zip(y).groupingBy { it.second.data[0] as Int }.eachCount().maxByOrNull { it.value }?.key ?: 0
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
//        if (np.any(np.isnan(X))) {
//            throw Exception("Input feature matrix X contains null values. Please provide a dataset without null values.")
//        }
//
//        if (np.any(np.isnan(y))) {
//            throw Exception("Target vector y contains null values. Please provide a target vector without null values.")
//        }
//
//        // تبدیل X به non-nullable ndarray
//        val nonNullX = X.mapNotNull { it.mapNotNull { value -> value.data[0] as Double } }.to_ndarray()
//
//        // تبدیل y به non-nullable ndarray
//        val nonNullY = y.mapNotNull { it.mapNotNull { value -> value.data[0] as Int } }.to_ndarray()

        // آموزش درخت‌های تصمیم
        for (i in 0 until numTrees) {
            val bootstrapSample = X.zip(y).shuffled().take(y.shape[0])
            val tree = DecisionTree(maxDepth, minSamplesSplit)
            tree.fit(
                X = bootstrapSample.map { it.first }.to_ndarray(),
                y = bootstrapSample.map { it.second }.to_ndarray()
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

//    println(df.shape)
//    println(df.drop(targetColumn).shape)
//    println(df[targetColumn].shape)
//    return
    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn],
        test_size = 0.2,
        random_state = 42
    )

    val randomForest = RandomForestClassifier(numTrees = 100, maxDepth = 10, minSamplesSplit = 2)
    randomForest.fit(X = dataSet.X_train, y = dataSet.y_train)

    var correctPredictions = 0
    for ((test_features, test_label) in dataSet.X_test.zip(dataSet.y_test)) {
        val prediction = randomForest.predict(test_features)
        if (prediction == test_label.data[0]) {
            correctPredictions++
        }
    }
    val accuracy = correctPredictions.toDouble() / dataSet.y_test.shape[0]
    println("Accuracy: $accuracy")
}