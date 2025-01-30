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

    // Load the Iris dataset
    val filePath = "D:\\repo\\AI4kt\\data\\classification\\iris.csv" // Update this path to your dataset location
    val df = read_csv(filePath)

    println(df.dtypes)

    // Define the target column
    val targetColumn = "species"

    df[targetColumn] = Series(df[targetColumn].map {
        if (it == "Iris-setosa") 0
        else if (it == "Iris-versicolor") 1
        else if (it == "Iris-virginica") 2
        else -1
    }.toMutableList())
    // Split the dataset into training and testing sets
    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Int>, // Target column is categorical (species names)
        test_size = 0.2,
        random_state = 42
    )

    // Build the deep learning model
    val model = Sequential(random)
        .addInput(4) // 4 features: sepal_length, sepal_width, petal_length, petal_width
        .addDense(10, ReLU()) // Hidden layer with 10 neurons
        .addDense(10, ReLU()) // Another hidden layer
        .addDense(3, Softmax()) // Output layer with 3 classes (Iris species)
        .setOptimizer(AdamOptimizer(0.001)) // Adam optimizer
        .setLossFunction(LossCategoricalCrossentropy()) // Loss function for multiclass classification
        .build()

    // One-hot encode the target labels
    val oneHot = OneHotEncoding()

    // Normalize the feature data
    val scaler = MinMaxScaler()
    dataSet["X_train"] = scaler.fitTransform(dataSet["X_train"] as D2Array<Double>)
    dataSet["X_test"] = scaler.transform(dataSet["X_test"] as D2Array<Double>)

    // Train the model
    model.fit(
        dataSet["X_train"] as D2Array<Double>,
        oneHot.transform(dataSet["y_train"] as D1Array<Int>),
        epochs = 100,
        batchSize = 32
    )

    // Make predictions
    val y_pred = model.predict(dataSet["X_test"] as D2Array<Double>) as D2Array<Double>
    val y_pred_class = y_pred.argmax(axis = 1) as D1Array<Int>

    // Calculate accuracy
    val accuracy = accuracy_score(dataSet["y_test"] as D1Array<Int>, y_pred_class)
    println("Accuracy: $accuracy")
}