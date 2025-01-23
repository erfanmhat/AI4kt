package io.ai4kt.ai4kt.fibonacci.sklearn.model_selection

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.sklearn.DataSet
import io.ai4kt.ai4kt.pandas.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.filter
import org.jetbrains.kotlinx.multik.ndarray.operations.filterMultiIndexed
import kotlin.random.Random


/**
 * Splits arrays or matrices into random train and test subsets.
 *
 * @param X The input features (DataFrame or List).
 * @param y The target labels (List).
 * @param test_size The proportion of the dataset to include in the test split (default = 0.2).
 * @param train_size The proportion of the dataset to include in the train split (default = complement of testSize).
 * @param random_state Seed for random number generation (default = null).
 * @param shuffle Whether to shuffle the data before splitting (default = true).
 * @param stratify If not null, data is split in a stratified fashion using this as the class labels (default = null).
 * @return A list containing train-test split of inputs: [X_train, X_test, y_train, y_test].
 */
fun train_test_split(
    X: DataFrame,
    y: Series<Int>,
    test_size: Double? = 0.2,
    train_size: Double? = null,
    random_state: Int? = null,
    shuffle: Boolean = true,
    stratify: List<Int>? = null
): DataSet {
    val X_D2Array = X.getValues(Double::class)
    val y_D1Array = y.getValues(Int::class)
    return train_test_split(
        X = X_D2Array,
        y = y_D1Array,
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
 * @param X The input features (DataFrame or List).
 * @param y The target labels (List).
 * @param test_size The proportion of the dataset to include in the test split (default = 0.2).
 * @param train_size The proportion of the dataset to include in the train split (default = complement of testSize).
 * @param random_state Seed for random number generation (default = null).
 * @param shuffle Whether to shuffle the data before splitting (default = true).
 * @param stratify If not null, data is split in a stratified fashion using this as the class labels (default = null).
 * @return A list containing train-test split of inputs: [X_train, X_test, y_train, y_test].
 */
fun train_test_split(
    X: D2Array<Double>,
    y: D1Array<Int>,
    test_size: Double? = 0.2,
    train_size: Double? = null,
    random_state: Int? = null,
    shuffle: Boolean = true,
    stratify: List<Int>? = null
): DataSet {
    require(X.shape[0] != 0) { "Input features (X) cannot be empty." }
    require(y.shape[0] != 0) { "Target labels (y) cannot be empty." }
    require(X.shape[0] == y.shape[0]) { "X and y must have the same length. X:${X.shape.contentToString()}, y:${y.shape.contentToString()}." }

    if (random_state != null) {
        Random(random_state)
    }

    val n_samples = X.shape[0]
    val (n_train, n_test) = calculate_train_test_sizes(n_samples, test_size, train_size)

    val indices = if (shuffle) {
        if (stratify != null) {
            stratified_shuffle_split(y, stratify, n_train, n_test)
        } else {
            (0 until n_samples).shuffled()
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

    // Create X_test by filtering rows based on test_indices
    val X_test = mk.zeros<Double>(test_indices.size, X.shape[1])
    for ((i, idx) in test_indices.withIndex()) {
        X_test[i] = X[idx]
    }

    // Create y_train by filtering elements based on train_indices
    val y_train = mk.zeros<Int>(train_indices.size)
    for ((i, idx) in train_indices.withIndex()) {
        y_train[i] = y[idx]
    }

    // Create y_test by filtering elements based on test_indices
    val y_test = mk.zeros<Int>(test_indices.size)
    for ((i, idx) in test_indices.withIndex()) {
        y_test[i] = y[idx]
    }

    return DataSet(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
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
private fun calculate_train_test_sizes(
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
private fun stratified_shuffle_split(
    y: D1Array<Int>,
    stratify: List<Int>,
    n_train: Int,
    n_test: Int
): List<Int> {
    val classCounts = stratify.groupingBy { it }.eachCount()
    val trainIndices = mutableListOf<Int>()
    val testIndices = mutableListOf<Int>()

    for ((cls, count) in classCounts) {
        // جمع‌آوری اندیس‌های مربوط به کلاس فعلی
        val clsIndices = mutableListOf<Int>()
        for (i in 0 until y.size) {
            if (y[i] == cls) {
                clsIndices.add(i)
            }
        }

        // شافل کردن اندیس‌های کلاس
        clsIndices.shuffle()

        // محاسبه تعداد نمونه‌های train و test برای این کلاس
        val n_train_cls = (count * n_train.toDouble() / y.size).toInt()
        val n_test_cls = (count * n_test.toDouble() / y.size).toInt()

        // اضافه کردن اندیس‌ها به train و test
        trainIndices.addAll(clsIndices.subList(0, n_train_cls))
        testIndices.addAll(clsIndices.subList(n_train_cls, n_train_cls + n_test_cls))
    }

    // ترکیب و شافل نهایی اندیس‌ها
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

    val (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.2, random_state = 42)

    println("X_train: $X_train")
    println("X_test: $X_test")
    println("y_train: $y_train")
    println("y_test: $y_test")
}
