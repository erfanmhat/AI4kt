package io.ai4kt.ai4kt.fibonacci

import io.ai4kt.ai4kt.fibonacci.pandas.Series
import io.ai4kt.ai4kt.fibonacci.pandas.read_csv
import io.ai4kt.ai4kt.fibonacci.sklearn.metrics.accuracy_score
import io.ai4kt.ai4kt.fibonacci.sklearn.model_selection.train_test_split
import io.ai4kt.ai4kt.fibonacci.tensorflow.OneHotEncoding
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Softmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.models.DeepLearningModel
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.AdamOptimizer
import kotlin.random.Random

fun main() {
    val random = Random(42)

    val filePath = "D:\\repo\\AI4kt\\data\\breast_cancer.csv"
    val df = read_csv(filePath)

    val targetColumn = "target"

    val dataSet = train_test_split(
        df.drop(targetColumn),
        df[targetColumn] as Series<Int>,
        test_size = 0.2,
        random_state = 42
    )

    val model = DeepLearningModel(random)
        .addInputLayer(30)
        .addDenseLayer(50, ReLU())
//        .addDenseLayer(50, ReLU())
        .addDenseLayer(2, Softmax())
        .setOptimizer(AdamOptimizer(0.001))
        .setLossFunction()
        .build()
    val oneHot = OneHotEncoding()

//    println(oneHot.transform(dataSet.y_train))
    model.fit(dataSet.X_train, oneHot.transform(dataSet.y_train), epochs = 100, batchSize = 32)


    val y_pred = model.predict(dataSet.X_test)
//    println(y_pred)
//    println(oneHot.inverseTransform(y_pred))
    val accuracy = accuracy_score(dataSet.y_test, oneHot.inverseTransform(y_pred))
    println("Accuracy: $accuracy")
}