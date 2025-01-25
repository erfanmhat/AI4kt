package io.ai4kt.ai4kt.fibonacci.tensorflow.models

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Activation
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Softmax
import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer
import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.InputLayer
import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.Layer
import io.ai4kt.ai4kt.fibonacci.tensorflow.loss.Loss
import io.ai4kt.ai4kt.fibonacci.tensorflow.loss.LossCategoricalCrossentropy
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.GradientDescentOptimizer
import io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers.Optimizer
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.random.Random

class DeepLearningModel(
    var random: Random
) {
    val layers = mutableListOf<Layer>()
    lateinit var optimizers: MutableList<Optimizer>
    lateinit var loss: Loss

    fun setRandom(random: Random): DeepLearningModel {
        this.random = random
        return this
    }

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
        layers.add(DNNLayer(nInputs, nNeurons, random, activation))
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
        val loss = mk.math.sum(loss.forward(output, yTrue))
        print("\r")
        print("Loss: $loss")
//        runBlocking {
//            delay(500)
//        }
        // Backward pass
        val dvalues = this.loss.backward(output, yTrue)
        backward(dvalues)

        // Update weights and biases
        for ((layer, optimizer) in layers.zip(optimizers)) {
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
            println()
            println("Epoch $epoch/$epochs")
            for (startIdx in 0 until nSamples step batchSize) {
                val endIdx = minOf(startIdx + batchSize, nSamples)
                val XBatch = X[startIdx until endIdx] as D2Array<Double>
                val yBatch = y[startIdx until endIdx] as D2Array<Double>
                // Perform a training step on the batch
                trainStep(XBatch, yBatch)
            }
        }
        println()
    }

    // Predict function
    fun predict(X: D2Array<Double>): D2Array<Double> {
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