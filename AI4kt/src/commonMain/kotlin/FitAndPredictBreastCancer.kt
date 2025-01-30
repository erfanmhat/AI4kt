import pandas.Series
import pandas.read_csv
import sklearn.MinMaxScaler
import sklearn.metrics.accuracy_score
import sklearn.model_selection.train_test_split
import tensorflow.OneHotEncoding
import tensorflow.activations.ReLU
import tensorflow.activations.Softmax
import tensorflow.argmax
import tensorflow.loss.LossCategoricalCrossentropy
import tensorflow.models.Sequential
import tensorflow.optimizers.AdamOptimizer
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.random.Random

fun main() {
    val random = Random(42)

    val filePath = "D:\\repo\\AI4kt\\data\\classification\\breast_cancer.csv"
    val df = read_csv(filePath)

    val targetColumn = "target"

    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Int>,
        test_size = 0.2,
        random_state = 42
    )

    val model = Sequential(random)
        .addInput(30)
        .addDense(25, ReLU())
        .addDense(25, ReLU())
        .addDense(25, ReLU())
        .addDense(2, Softmax())
        .setOptimizer(AdamOptimizer(0.001))
//        .setOptimizer(GradientDescentOptimizer(0.001))
        .setLossFunction(LossCategoricalCrossentropy())
        .build()
    val oneHot = OneHotEncoding()

    val scaler = MinMaxScaler()
    dataSet["X_train"] = scaler.fitTransform(dataSet["X_train"] as D2Array<Double>) as Any
    dataSet["X_test"] = scaler.transform(dataSet["X_test"] as D2Array<Double>)
    model.fit(
        dataSet["X_train"] as D2Array<Double>,
        oneHot.transform(dataSet["y_train"] as D1Array<Int>),
        epochs = 100,
        batchSize = 32
    )


    val y_pred = model.predict(dataSet["X_test"] as D2Array<Double>) as D2Array<Double>
    val y_pred_class = y_pred.argmax(axis = 1) as D1Array<Int>
    val accuracy = accuracy_score(dataSet["y_test"] as D1Array<Int>, y_pred_class)
    println("Accuracy: $accuracy")
}