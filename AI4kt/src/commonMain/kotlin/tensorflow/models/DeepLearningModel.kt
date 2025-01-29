package tensorflow.models

import tensorflow.activations.Activation
import tensorflow.layers.*
import tensorflow.loss.Loss
import tensorflow.optimizers.Optimizer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import tensorflow.activations.ReLU
import tensorflow.activations.Softmax
import tensorflow.loss.LossCategoricalCrossentropy
import tensorflow.optimizers.GradientDescentOptimizer
import kotlin.random.Random
import tensorflow.get

class DeepLearningModel(
    var random: Random
) {
    val layers = mutableListOf<Layer>()
    lateinit var optimizers: MutableList<Optimizer>
    lateinit var loss: Loss
    private var epochLosses = mutableListOf<Double>()

    fun setRandom(random: Random): DeepLearningModel {
        this.random = random
        return this
    }

    // Builder methods
    fun addInputLayer(vararg inputShape: Int): DeepLearningModel {
        layers.add(InputLayer(*inputShape))
        return this
    }

    fun addDenseLayer(nNeurons: Int, activation: Activation? = null): DeepLearningModel {
        val nInputs = when (val lastLayer = layers.lastOrNull()) {
            is InputLayer -> lastLayer.inputShape.last() // Last dimension of input shape
            is DNNLayer -> lastLayer.weights.shape[1]
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(DNNLayer(nInputs, nNeurons, random, activation))
        return this
    }

    fun addLayer(layer: Layer): DeepLearningModel {
        layers.add(layer)
        return this
    }

    fun setOptimizer(optimizer: Optimizer): DeepLearningModel {
        optimizers = mutableListOf()
        for (layer in layers) {
            this.optimizers.add(optimizer.copy())
        }
        return this
    }

    fun setLossFunction(loss: Loss): DeepLearningModel {
        this.loss = loss
        return this
    }

    fun build(): DeepLearningModel {
        return this
    }

    // Forward pass
    fun forward(inputs: NDArray<Double, *>): D2Array<Double> {
        var output = inputs
        for (layer in layers) {
            output = layer.forward(output)
        }
        return output as D2Array<Double>
    }

    // Backward pass
    fun backward(dvalues: NDArray<Double, *>) {
        var grad = dvalues // Start with the initial gradients

        // Iterate through layers in reverse order
        for (layer in layers.reversed()) {
            grad = layer.backward(grad)
        }
    }

    // Train step
    fun trainStep(inputs: NDArray<Double, *>, yTrue: D2Array<Double>) {
        // Forward pass
        val output = forward(inputs)

        // Compute loss
        val trainStepLoss = mk.math.sum(loss.forward(output, yTrue))
        epochLosses.add(trainStepLoss)

        print("\r")
        print("Loss: ${epochLosses.average()}")

        // Backward pass
        val dvalues = loss.backward(output, yTrue)
        backward(dvalues)

        // Update weights and biases
        for ((layer, optimizer) in layers.zip(optimizers)) {
            if (layer is TrainableLayer) {
                optimizer.update(layer)
            }
        }
    }

    // Fit function
    fun fit(
        X: NDArray<Double, *>,
        y: D2Array<Double>,
        epochs: Int,
        batchSize: Int
    ) {
        val nSamples = X.shape[0]
        for (epoch in 1..epochs) {
            epochLosses = mutableListOf()
            println()
            println("Epoch $epoch/$epochs")
            for (startIdx in 0 until nSamples step batchSize) {
                val endIdx = minOf(startIdx + batchSize, nSamples)
                val XBatch = X[startIdx until endIdx]
                val yBatch = y[startIdx until endIdx]
                // Perform a training step on the batch
                trainStep(XBatch, yBatch as D2Array<Double>)
            }
        }
        println()
    }

    // Predict function
    fun predict(X: NDArray<Double, *>): NDArray<Double, *> {
        return forward(X)
    }
}

fun main() {
    val random = Random(42)
    // Create a model using the builder pattern
    val model = DeepLearningModel(random)
        .addInputLayer(3) // Input layer with 3 features
        .addDenseLayer(30, ReLU()) // Hidden layer with 5 neurons and ReLU activation
        .addDenseLayer(3, Softmax()) // Output layer with 2 neurons and Softmax activation
        .setOptimizer(GradientDescentOptimizer(0.001)) // Set optimizer with learning rate 0.01
        .setLossFunction(LossCategoricalCrossentropy()) // Set loss function
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


    model.fit(inputs, yTrue, epochs = 100, batchSize = 1)
}