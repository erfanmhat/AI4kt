package sklearn.model_selection

import pandas.Series
import pandas.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random

/**
 * Splits arrays or matrices into random train and test subsets.
 *
 * @param X The input features (DataFrame or List).
 * @param y The target labels (Series or D1Array).
 * @param test_size The proportion of the dataset to include in the test split (default = 0.2).
 * @param train_size The proportion of the dataset to include in the train split (default = complement of testSize).
 * @param random_state Seed for random number generation (default = null).
 * @param shuffle Whether to shuffle the data before splitting (default = true).
 * @param stratify If not null, data is split in a stratified fashion using this as the class labels (default = null).
 * @return A map containing train-test split of inputs: [X_train, X_test, y_train, y_test].
 */
inline fun <reified T : Number> train_test_split(
    X: DataFrame,
    y: Series<T>,
    test_size: Double? = 0.2,
    train_size: Double? = null,
    random_state: Int? = null,
    shuffle: Boolean = true,
    stratify: List<Int>? = null
): MutableMap<String, Any> {
    val X_D2Array = X.getValues(Double::class)
    val y_D1Array = when (T::class) {
        Int::class -> y.getValues(Int::class)
        Double::class -> y.getValues(Double::class)
        else -> throw IllegalArgumentException("Unsupported type for y: ${T::class}")
    }
    return train_test_split(
        X = X_D2Array,
        y = y_D1Array as D1Array<T>,
        test_size = test_size,
        train_size = train_size,
        random_state = random_state,
        shuffle = shuffle,
        stratify = stratify
    )
}

/**
 * Splits arrays or matrices into random train and test subsets.
 *
 * @param X The input features (D2Array<Double>).
 * @param y The target labels (D1Array<T>).
 * @param test_size The proportion of the dataset to include in the test split (default = 0.2).
 * @param train_size The proportion of the dataset to include in the train split (default = complement of testSize).
 * @param random_state Seed for random number generation (default = null).
 * @param shuffle Whether to shuffle the data before splitting (default = true).
 * @param stratify If not null, data is split in a stratified fashion using this as the class labels (default = null).
 * @return A map containing train-test split of inputs: [X_train, X_test, y_train, y_test].
 */
inline fun <reified T : Number> train_test_split(
    X: D2Array<Double>,
    y: D1Array<T>,
    test_size: Double? = 0.2,
    train_size: Double? = null,
    random_state: Int? = null,
    shuffle: Boolean = true,
    stratify: List<Int>? = null
): MutableMap<String, Any> {
    require(X.shape[0] != 0) { "Input features (X) cannot be empty." }
    require(y.shape[0] != 0) { "Target labels (y) cannot be empty." }
    require(X.shape[0] == y.shape[0]) { "X and y must have the same length. X:${X.shape.contentToString()}, y:${y.shape.contentToString()}." }

    var random: Random? = null
    if (random_state != null) {
        random = Random(random_state)
    }

    val n_samples = X.shape[0]
    val (n_train, n_test) = calculate_train_test_sizes(n_samples, test_size, train_size)

    val indices = if (shuffle) {
        if (stratify != null) {
            stratified_shuffle_split(y, stratify, n_train, n_test)
        } else {
            if (random != null) (0 until n_samples).shuffled(random)
            else (0 until n_samples).shuffled()
        }
    } else {
        if (stratify != null) {
            throw IllegalArgumentException("Stratified split is not supported when shuffle=false.")
        }
        (0 until n_samples).toList()
    }

    val train_indices = indices.subList(0, n_train)
    val test_indices = indices.subList(n_train, n_train + n_test)

    val X_train = mk.zeros<Double>(train_indices.size, X.shape[1])
    for ((i, idx) in train_indices.withIndex()) {
        X_train[i] = X[idx]
    }

    val X_test = mk.zeros<Double>(test_indices.size, X.shape[1])
    for ((i, idx) in test_indices.withIndex()) {
        X_test[i] = X[idx]
    }

    val y_train = when (T::class) {
        Int::class -> mk.zeros<Int>(train_indices.size) as D1Array<T>
        Double::class -> mk.zeros<Double>(train_indices.size) as D1Array<T>
        else -> throw IllegalArgumentException("Unsupported type for y: ${T::class}")
    }
    for ((i, idx) in train_indices.withIndex()) {
        y_train[i] = y[idx]
    }

    val y_test = when (T::class) {
        Int::class -> mk.zeros<Int>(test_indices.size) as D1Array<T>
        Double::class -> mk.zeros<Double>(test_indices.size) as D1Array<T>
        else -> throw IllegalArgumentException("Unsupported type for y: ${T::class}")
    }
    for ((i, idx) in test_indices.withIndex()) {
        y_test[i] = y[idx]
    }

    return mutableMapOf(
        "X_train" to X_train,
        "X_test" to X_test,
        "y_train" to y_train,
        "y_test" to y_test
    )
}

/**
 * Calculates the sizes of train and test sets based on the given proportions.
 *
 * @param n_samples Total number of samples.
 * @param test_size The proportion of the dataset to include in the test split.
 * @param train_size The proportion of the dataset to include in the train split.
 * @return A pair containing the number of train and test samples.
 */
fun calculate_train_test_sizes(
    n_samples: Int,
    test_size: Double?,
    train_size: Double?
): Pair<Int, Int> {
    require(test_size == null || test_size > 0.0) { "test_size must be greater than 0." }
    require(train_size == null || train_size > 0.0) { "train_size must be greater than 0." }

    val testSize = test_size ?: 0.2
    val trainSize = train_size ?: (1.0 - testSize)

    require(testSize + trainSize <= 1.0) { "The sum of test_size and train_size must be less than or equal to 1.0." }

    val n_test = (n_samples * testSize).toInt()
    val n_train = (n_samples * trainSize).toInt()

    return Pair(n_train, n_test)
}

/**
 * Performs a stratified shuffle split.
 *
 * @param y The target labels.
 * @param stratify The class labels for stratification.
 * @param n_train Number of training samples.
 * @param n_test Number of test samples.
 * @return A list of shuffled indices.
 */
fun <T : Number> stratified_shuffle_split(
    y: D1Array<T>,
    stratify: List<Int>,
    n_train: Int,
    n_test: Int
): List<Int> {
    val classCounts = stratify.groupingBy { it }.eachCount()
    val trainIndices = mutableListOf<Int>()
    val testIndices = mutableListOf<Int>()

    for ((cls, count) in classCounts) {
        val clsIndices = mutableListOf<Int>()
        for (i in 0 until y.size) {
            if (y[i].toInt() == cls) {
                clsIndices.add(i)
            }
        }

        clsIndices.shuffle()

        val n_train_cls = (count * n_train.toDouble() / y.size).toInt()
        val n_test_cls = (count * n_test.toDouble() / y.size).toInt()

        trainIndices.addAll(clsIndices.subList(0, n_train_cls))
        testIndices.addAll(clsIndices.subList(n_train_cls, n_train_cls + n_test_cls))
    }

    return (trainIndices + testIndices).shuffled()
}

fun main() {
    val X = mk.ndarray(
        mk[
            mk[1.0, 2.0],
            mk[2.0, 3.0],
            mk[3.0, 4.0],
            mk[4.0, 5.0],
            mk[5.0, 6.0]
        ]
    )
    val y = mk.ndarray(mk[0, 1, 0, 1, 0])

    val dataSet = train_test_split(X, y, test_size = 0.2, random_state = 42)

    println("X_train: " + dataSet["X_train"])
    println("X_test: " + dataSet["X_test"])
    println("y_train: " + dataSet["y_train"])
    println("y_test: " + dataSet["y_test"])
}
