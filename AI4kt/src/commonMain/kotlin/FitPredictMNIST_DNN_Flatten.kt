import sklearn.metrics.accuracy_score
import tensorflow.OneHotEncoding
import tensorflow.activations.ReLU
import tensorflow.activations.Softmax
import tensorflow.argmax
import tensorflow.loss.LossCategoricalCrossentropy
import tensorflow.models.Sequential
import tensorflow.optimizers.AdamOptimizer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import tensorflow.layers.Conv2D
import tensorflow.layers.Flatten
import kotlin.random.Random

fun main() {
    val random = Random(42)
    val X_trainPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\train-images.idx3-ubyte"
    val y_trainPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\train-labels.idx1-ubyte"
    val X_testPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\t10k-images.idx3-ubyte"
    val y_testPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\t10k-labels.idx1-ubyte"
    var X_train = mk.ndarray(readIDX3Images(X_trainPath))
    val y_train = mk.ndarray(readIDX1Labels(y_trainPath))
    var X_test = mk.ndarray(readIDX3Images(X_testPath))
    val y_test = mk.ndarray(readIDX1Labels(y_testPath))

    X_train = X_train.map { it / 255.0 }
    X_test = X_test.map { it / 255.0 }

    val model = Sequential(
        batchSize = 32,
        random = random
    )
        .addInput(28, 28)
        .addFlatten()
        .addDense(128, ReLU())
        .addDense(64, ReLU())  // Another hidden layer with 64 neurons
        .addDense(10, Softmax()) // Output layer with 10 classes (digits 0-9)
        .setOptimizer(AdamOptimizer(0.001)) // Adam optimizer
        .setLossFunction(LossCategoricalCrossentropy()) // Loss function for multiclass classification
        .build()
    val oneHot = OneHotEncoding()
    model.fit(
        X_train,
        oneHot.transform(y_train),
        epochs = 100
    )

    val y_pred = model.predict(X_test) as D2Array<Double>
    val y_pred_class = y_pred.argmax(axis = 1) as D1Array<Int>

    val accuracy = accuracy_score(y_test, y_pred_class)
    println("Accuracy: $accuracy")
}