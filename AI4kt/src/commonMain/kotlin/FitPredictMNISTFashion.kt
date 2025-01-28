package io.ai4kt.ai4kt.fibonacci

import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.MinMaxScaler
import io.ai4kt.ai4kt.fibonacci.sklearn.metrics.accuracy_score
import io.ai4kt.ai4kt.fibonacci.tensorflow.OneHotEncoding
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Softmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.argmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.loss.LossCategoricalCrossentropy
import io.ai4kt.ai4kt.fibonacci.tensorflow.models.DeepLearningModel
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.AdamOptimizer
import io.ai4kt.ai4kt.pandas.DataFrame
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.random.Random

fun main() {
    val random = Random(42)

    val filePathTrain =
        "D:\\repo\\AI4kt\\data\\classification\\MNIST_FASHION\\fashion-mnist_train.csv"
    val filePathTest =
        "D:\\repo\\AI4kt\\data\\classification\\MNIST_FASHION\\fashion-mnist_test.csv"
    val df_train = read_csv(filePathTrain)
    val df_test = read_csv(filePathTest)

    val targetColumn = "label"
    var X_train = df_train.drop(targetColumn)
    val y_train = df_train[targetColumn].getValues(Int::class)
    var X_test = df_test.drop(targetColumn)
    val y_test = df_test[targetColumn].getValues(Int::class)

    for (column in X_train.columns) {
        if (X_train[column].max() == 0.0) {
            X_train = X_train.drop(column)
            X_test = X_test.drop(column)
        }
    }

    val scaler = MinMaxScaler()
    val X_train_D2 = scaler.fitTransform(X_train.getValues(Double::class))
    val X_test_D2 = scaler.transform(X_test.getValues(Double::class))

    val model = DeepLearningModel(random)
        .addInputLayer(X_train.shape[1])
        .addDenseLayer(128, ReLU()) // Hidden layer with 128 neurons
        .addDenseLayer(64, ReLU())  // Another hidden layer with 64 neurons
        .addDenseLayer(10, Softmax()) // Output layer with 10 classes (digits 0-9)
        .setOptimizer(AdamOptimizer(0.001)) // Adam optimizer
        .setLossFunction(LossCategoricalCrossentropy()) // Loss function for multiclass classification
        .build()
    val oneHot = OneHotEncoding()
    model.fit(
        X_train_D2,
        oneHot.transform(y_train),
        epochs = 10,
        batchSize = 32
    )

    val y_pred = model.predict(X_test_D2)
    val y_pred_class = y_pred.argmax(axis = 1) as D1Array<Int>

    val accuracy = accuracy_score(y_test, y_pred_class)
    println("Accuracy: $accuracy")
}