package io.ai4kt.ai4kt.fibonacci

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.ensemble.RandomForestClassifier
import io.ai4kt.ai4kt.fibonacci.sklearn.metrics.accuracy_score
import io.ai4kt.ai4kt.fibonacci.sklearn.model_selection.train_test_split
import io.ai4kt.ai4kt.fibonacci.tensorflow.OneHotEncoding
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Softmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.models.DeepLearningModel

fun main() {

    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
    val df = read_csv(filePath)

    val targetColumn = "target"

    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Int>,
        test_size = 0.2,
        random_state = 42
    )

    val model = DeepLearningModel()
        .addInputLayer(30) // Input layer with 3 features
        .addDenseLayer(50, ReLU()) // Hidden layer with 5 neurons and ReLU activation
        .addDenseLayer(2, Softmax()) // Output layer with 2 neurons and Softmax activation
        .setOptimizer(0.01) // Set optimizer with learning rate 0.01
        .setLossFunction() // Set loss function
        .build()
    val oneHot = OneHotEncoding()

    model.fit(dataSet.X_train, oneHot.transform(dataSet.y_train), epochs = 100, batchSize = 32)


    val y_pred = model.predict(dataSet.X_test)
    val accuracy = accuracy_score(dataSet.y_test, oneHot.inverseTransform(y_pred))
    println("Accuracy: $accuracy")
}