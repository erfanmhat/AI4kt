import pandas.Series
import pandas.asSeries
import pandas.read_csv
import sklearn.MinMaxScaler
import sklearn.metrics.mean_squared_error
import sklearn.model_selection.train_test_split
import tensorflow.OneHotEncodingSeries
import tensorflow.activations.ReLU
import tensorflow.loss.LossMeanSquaredError
import tensorflow.models.Sequential
import tensorflow.optimizers.AdamOptimizer
import pandas.asDataFrame
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.random.Random

fun main() {
    // todo fix regression bad prediction
    val random = Random(42)

    // Load the California Housing dataset
    val filePath = "D:\\repo\\AI4kt\\data\\California_Housing_Dataset.csv" // Update this path to your dataset location
    var df = read_csv(filePath)

    val oneHotEncoding = OneHotEncodingSeries()
    val df2 = oneHotEncoding.transform(df["ocean_proximity"]).asDataFrame("ocean_proximity")

    df = df.concat(df2, axis = 1).drop(listOf("ocean_proximity", "total_bedrooms"))

    println(df.head())
    // Define the target column
    val targetColumn = "median_house_value" // Median house value for California districts

//    for (col in df.data.keys) {
//        for (i in 0..<df.shape[0]) {
//            if (df.data[col]!![i].toString().isEmpty()) {
//                df.data[col]!![i] = 0.0//TODO find better way to do that
//            }
//        }
//    }
    df[targetColumn] = (
            (df[targetColumn].getValues(Double::class) - df[targetColumn].min()) /
                    (df[targetColumn].max() - df[targetColumn].min())
            ).asD1Array().asSeries()

    // Split the dataset into training and testing sets
    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Double>, // Target column is continuous (regression)
        test_size = 0.2,
        random_state = 42
    )

    // Build the deep learning model
    val model = Sequential(random)
        .addInput(12) // 8 features in the California Housing dataset
//        .addDense(1024, ReLU()) // Hidden layer with 100 neurons
//        .addDense(512, ReLU()) // Hidden layer with 100 neurons
//        .addDense(256, ReLU()) // Hidden layer with 100 neurons
//        .addDense(128, ReLU()) // Hidden layer with 100 neurons
        .addDense(64, ReLU()) // Another hidden layer
        .addDense(32, ReLU()) // Another hidden layer
        .addDense(1) // Output layer with 1 neuron (regression task)
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
        epochs = 14,
        batchSize = 32
    )

    // Make predictions
    val y_pred = model.predict(dataSet["X_test"] as D2Array<Double>)
    println(y_pred.toList().zip((dataSet["y_test"] as D1Array<Double>).toList()))

    // Calculate Mean Squared Error (MSE)
    val mse = mean_squared_error(dataSet["y_test"] as D1Array<Double>, y_pred.flatten() as D1Array<*>)
    println("Mean Squared Error: $mse")
}