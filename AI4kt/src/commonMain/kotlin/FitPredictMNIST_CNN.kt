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
import tensorflow.KernelSize
import tensorflow.layers.Conv2D
import tensorflow.layers.MaxPooling2D
import kotlin.random.Random

fun main() {
    val random = Random(42)
    val X_trainPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\train-images.idx3-ubyte"
    val y_trainPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\train-labels.idx1-ubyte"
    val X_testPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\t10k-images.idx3-ubyte"
    val y_testPath = "D:\\repo\\AI4kt\\data\\classification\\MNIST\\t10k-labels.idx1-ubyte"
    val X_train = mk.ndarray(readIDX3Images(X_trainPath))
    val y_train = mk.ndarray(readIDX1Labels(y_trainPath))
    val X_test = mk.ndarray(readIDX3Images(X_testPath))
    val y_test = mk.ndarray(readIDX1Labels(y_testPath))

    var X_train4D = X_train.reshape(X_train.shape[0], 28, 28, 1)
    var X_test4D = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train4D = X_train4D.map { it / 255.0 }
    X_test4D = X_test4D.map { it / 255.0 }

    val batchSize = 32
    val model = Sequential(
        batchSize = batchSize,
        random = random
    )
        .addInput(28, 28, 1)
        .add(
            Conv2D(
                inputShape = intArrayOf(batchSize, 28, 28, 1),
                filters = 32,  // Increased number of filters
                kernelSize = KernelSize(3, 3),
                strides = intArrayOf(1, 1),
                padding = "same",
                random = random,
                activation = ReLU()
            )
        )
        .addMaxPooling2D(
                poolSize = Pair(2, 2),
                strides = Pair(2, 2),
                padding = "valid"
        )
        .add(
            Conv2D(
                inputShape = intArrayOf(batchSize, 28, 28, 32),
                filters = 64,  // Increased number of filters
                kernelSize = KernelSize(3, 3),
                strides = intArrayOf(1, 1),
                padding = "same",
                random = random,
                activation = ReLU()
            )
        )
        .addMaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid"
        )
        .add(
            Conv2D(
                filters = 128,  // Increased number of filters
                kernelSize = KernelSize(3, 3),
                strides = intArrayOf(1, 1),
                padding = "same",
                random = random,
                activation = ReLU(),
                inputShape = intArrayOf(batchSize, 28, 28, 64)
            )
        )
        .addFlatten()
        .addDense(256, ReLU())  // Increased number of units
//        .addDropout(0.5)  // Added dropout layer
        .addDense(128, ReLU())  // Added another dense layer
//        .addDropout(0.5)  // Added dropout layer
        .addDense(10, Softmax())
        .setOptimizer(AdamOptimizer(0.001))
        .setLossFunction(LossCategoricalCrossentropy())
        .build()

    val oneHot = OneHotEncoding()
    model.fit(
        X_train4D,
        oneHot.transform(y_train),
        epochs = 20  // Increased number of epochs
    )

    val y_pred = model.predict(X_test4D) as D2Array<Double>
    val y_pred_class = y_pred.argmax(axis = 1) as D1Array<Int>

    val accuracy = accuracy_score(y_test, y_pred_class)
    println("Accuracy: $accuracy")
}