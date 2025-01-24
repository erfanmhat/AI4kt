package io.ai4kt.ai4kt.fibonacci.tensorflow.models

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Activation
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Softmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer
import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.InputLayer
import io.ai4kt.ai4kt.fibonacci.tensorflow.loss.LossCategoricalCrossentropy
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.GradientDescentOptimizer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get

class DeepLearningModel {
    val layers = mutableListOf<Any>()
    lateinit var optimizer: GradientDescentOptimizer
    lateinit var lossFunction: LossCategoricalCrossentropy

    // Builder methods
    fun addInputLayer(nInputs: Int): DeepLearningModel {
        layers.add(InputLayer(nInputs))
        return this
    }

    fun addDenseLayer(nNeurons: Int, activation: Activation? = null): DeepLearningModel {
        val nInputs = when (val lastLayer = layers.lastOrNull()) {
            is InputLayer -> lastLayer.nInputs
            is DNNLayer -> lastLayer.weights.shape[1]
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(DNNLayer(nInputs, nNeurons, activation))
        return this
    }

    fun setOptimizer(learningRate: Double): DeepLearningModel {
        optimizer = GradientDescentOptimizer(learningRate)
        return this
    }

    fun setLossFunction(): DeepLearningModel {
        lossFunction = LossCategoricalCrossentropy()
        return this
    }

    fun build(): DeepLearningModel {
        return this
    }

    // Forward pass
    fun forward(inputs: D2Array<Double>): D2Array<Double> {
        var output = inputs
        for (layer in layers) {
            output = when (layer) {
                is InputLayer -> layer.forward(output)
                is DNNLayer -> layer.forward(output)
                else -> throw IllegalArgumentException("Invalid layer type: ${layer::class.simpleName}")
            }
        }
        return output
    }

    // Backward pass
    fun backward(dvalues: D2Array<Double>) {
        var grad = dvalues // Start with the initial gradients

        // Iterate through layers in reverse order
        for (layer in layers.reversed()) {
            grad = when (layer) {
                is InputLayer -> grad
                is DNNLayer -> layer.backward(grad) // Update gradients for DNNLayer
                else -> throw IllegalArgumentException("Unsupported layer type: ${layer::class.simpleName}")
            }
        }
    }

    // Train step
    fun trainStep(inputs: D2Array<Double>, yTrue: D2Array<Double>) {
        // Forward pass
        val output = forward(inputs)

        // Compute loss
        val loss = lossFunction.calculate(output, yTrue)
        println("Loss: $loss")

        // Backward pass
        val dvalues = lossFunction.backward(output, yTrue)
        backward(dvalues)

        // Update weights and biases
        for (layer in layers) {
            if (layer is DNNLayer) {
                optimizer.update(layer)
            }
        }
    }

    // Fit function
    fun fit(
        X: D2Array<Double>,
        y: D2Array<Double>,
        epochs: Int,
        batchSize: Int
    ) {
        val nSamples = X.shape[0]
        for (epoch in 1..epochs) {
            println("Epoch $epoch/$epochs")
            for (startIdx in 0 until nSamples step batchSize) {
                val endIdx = minOf(startIdx + batchSize, nSamples)
                val XBatch = X[startIdx until endIdx] as D2Array<Double>
                val yBatch = y[startIdx until endIdx] as D2Array<Double>

                // Perform a training step on the batch
                trainStep(XBatch, yBatch)
            }
        }
    }

    // Predict function
    fun predict(X: D2Array<Double>): D2Array<Double> {
        return forward(X)
    }
}

fun main() {
    // Create a model using the builder pattern
    val model = DeepLearningModel()
        .addInputLayer(3) // Input layer with 3 features
        .addDenseLayer(30, ReLU()) // Hidden layer with 5 neurons and ReLU activation
        .addDenseLayer(3, Softmax()) // Output layer with 2 neurons and Softmax activation
        .setOptimizer(0.01) // Set optimizer with learning rate 0.01
        .setLossFunction() // Set loss function
        .build()

    // Example input data (2 samples, 3 features each)
    val inputs: D2Array<Double> = mk.ndarray(
        listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        )
    )

    // Example target data (2 samples, one-hot encoded)
    val yTrue: D2Array<Double> = mk.ndarray(
        listOf(
            listOf(1.0, 0.0, 0.0),
            listOf(0.0, 1.0, 0.0),
            listOf(0.0, 0.0, 1.0)
        )
    )

    // Train the model for one step
    for (i in 0..1000) {
        model.trainStep(inputs, yTrue)
    }
}