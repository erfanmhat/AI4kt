import pandas.Series
import pandas.read_csv
import sklearn.MinMaxScaler
import sklearn.metrics.mean_squared_error
import sklearn.model_selection.train_test_split
import tensorflow.activations.ReLU
import tensorflow.loss.LossMeanSquaredError
import tensorflow.models.DeepLearningModel
import tensorflow.optimizers.AdamOptimizer
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.random.Random

fun main() {
    val random = Random(42)

    // Load the Boston Housing dataset
    val filePath = "D:\\repo\\AI4kt\\data\\Boston_Housing_Dataset.csv" // Update this path to your dataset location
    var df = read_csv(filePath)
    df = df.drop("Index")

    // Define the target column
    val targetColumn = "PRICE" // Median value of owner-occupied homes (in $1000s)

    // Split the dataset into training and testing sets
    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Double>, // Target column is continuous (regression)
        test_size = 0.2,
        random_state = 42
    )

    // Build the deep learning model
    val model = DeepLearningModel(random)
        .addInputLayer(13) // 13 features in the Boston Housing dataset
        .addDenseLayer(100, ReLU()) // Hidden layer with 10 neurons
        .addDenseLayer(50, ReLU()) // Another hidden layer
        .addDenseLayer(20, ReLU()) // Another hidden layer
        .addDenseLayer(1) // Output layer with 1 neuron (regression task)
        .setOptimizer(AdamOptimizer(0.001)) // Adam optimizer
        .setLossFunction(LossMeanSquaredError()) // Loss function for regression
        .build()

    // Normalize the feature data
    val scaler = MinMaxScaler()
    dataSet["X_train"] = scaler.fitTransform(dataSet["X_train"] as D2Array<Double>)
    dataSet["X_test"] = scaler.transform(dataSet["X_test"] as D2Array<Double>)

    // Train the model
    val y_train = dataSet["y_train"] as D1Array<Double>
    model.fit(
        dataSet["X_train"] as D2Array<Double>,
        y_train.reshape(y_train.shape[0], 1),
        epochs = 140,
        batchSize = 32
    )

    // Make predictions
    val y_pred = model.predict(dataSet["X_test"] as D2Array<Double>)

    // Calculate Mean Squared Error (MSE)
    val mse = mean_squared_error(dataSet["y_test"] as D1Array<Double>, y_pred.flatten() as D1Array<*>)
    println("Mean Squared Error: $mse")
}