package io.ai4kt.ai4kt.fibonacci

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.MinMaxScaler
import io.ai4kt.ai4kt.fibonacci.sklearn.metrics.mean_squared_error
import io.ai4kt.ai4kt.fibonacci.sklearn.model_selection.train_test_split
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.loss.LossMeanSquaredError
import io.ai4kt.ai4kt.fibonacci.tensorflow.models.DeepLearningModel
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.AdamOptimizer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.random.Random

fun main() {
    val random = Random(42)

    // Load the Boston Housing dataset
    val filePath = "D:\\repo\\AI4kt\\data\\Boston_Housing_Dataset.csv" // Update this path to your dataset location
    val df = read_csv(filePath)
    println(df.dtypes)
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
        .addDenseLayer(10, ReLU()) // Hidden layer with 10 neurons
        .addDenseLayer(10, ReLU()) // Another hidden layer
        .addDenseLayer(1) // Output layer with 1 neuron (regression task)
        .setOptimizer(AdamOptimizer(0.001)) // Adam optimizer
        .setLossFunction(LossMeanSquaredError()) // Loss function for regression
        .build()

    // Normalize the feature data
    val scaler = MinMaxScaler()
    dataSet["X_train"] = scaler.fitTransform(dataSet["X_train"] as D2Array<Double>)
    dataSet["X_test"] = scaler.transform(dataSet["X_test"] as D2Array<Double>)

    // Train the model
    model.fit(
        dataSet["X_train"] as D2Array<Double>,
        dataSet["y_train"] as D1Array<Double>, // No one-hot encoding needed for regression
        epochs = 100,
        batchSize = 32
    )

    // Make predictions
    val y_pred = model.predict(dataSet["X_test"] as D2Array<Double>)

    // Calculate Mean Squared Error (MSE)
    val mse = mean_squared_error(dataSet["y_test"] as D1Array<Double>, y_pred.flatten() as D1Array<*>)
    println("Mean Squared Error: $mse")
}