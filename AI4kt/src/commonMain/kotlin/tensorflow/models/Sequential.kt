package tensorflow.models

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.api.ndarray
import tensorflow.activations.Activation
import tensorflow.layers.*
import tensorflow.loss.Loss
import tensorflow.optimizers.Optimizer
import tensorflow.activations.ReLU
import tensorflow.activations.Softmax
import tensorflow.loss.LossCategoricalCrossentropy
import tensorflow.optimizers.GradientDescentOptimizer
import kotlin.random.Random
import tensorflow.get

class Sequential(
    val batchSize: Int,
    var random: Random
) {
    val layers = mutableListOf<Layer>()
    lateinit var optimizers: MutableList<Optimizer>
    lateinit var loss: Loss
    private var epochLosses = mutableListOf<Double>()
    var outputShape = intArrayOf()

    fun setRandom(random: Random): Sequential {
        this.random = random
        return this
    }

    // Builder methods
    fun addInput(vararg inputShape: Int): Sequential {
        layers.add(Input(*inputShape))
        return this
    }

    fun addFlatten(): Sequential {
        val flattenInputShape = when (val lastLayer = layers.lastOrNull()) {
            is Input -> lastLayer.inputShape
            is Conv2D -> lastLayer.outputShape
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(Flatten(batchSize, flattenInputShape))
        return this
    }

    fun addDense(nNeurons: Int, activation: Activation? = null): Sequential {
        val nInputs = when (val lastLayer = layers.lastOrNull()) {
            is Input -> lastLayer.inputShape.last() // Last dimension of input shape
            is Dense -> lastLayer.weights.shape[1]
            is Flatten -> lastLayer.outputShape[1]
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(Dense(nInputs, nNeurons, random, activation))
        return this
    }

    // Add a Convolutional Layer
    fun addConv2D(
        filters: Int,
        kernelSize: Pair<Int, Int>,
        padding: String = "valid",
        activation: Activation? = ReLU()
    ): Sequential {
        val inputShape = when (val lastLayer = layers.lastOrNull()) {
            is Input -> lastLayer.inputShape
            is Conv2D -> lastLayer.outputShape
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(
            Conv2D(
                filters = filters,
                kernelSize = kernelSize,
                padding = padding,
                activation = activation,
                inputShape = inputShape,
                random = random
            )
        )
        return this
    }

    // Add a Max Pooling Layer
    fun addMaxPooling2D(
        poolSize: Pair<Int, Int> = Pair(2, 2),
        strides: Pair<Int, Int> = Pair(2, 2),
        padding: String = "valid"
    ): Sequential {
        val inputShape = when (val lastLayer = layers.lastOrNull()) {
            is Input -> lastLayer.inputShape
            is Conv2D -> lastLayer.outputShape
            else -> throw IllegalArgumentException("Invalid layer type")
        }
        layers.add(MaxPooling2D(poolSize, strides, padding, inputShape))
        return this
    }

    fun add(layer: Layer): Sequential {
        layers.add(layer)
        return this
    }

    fun setOptimizer(optimizer: Optimizer): Sequential {
        optimizers = mutableListOf()
        for (layer in layers) {
            this.optimizers.add(optimizer.copy())
        }
        return this
    }

    fun setLossFunction(loss: Loss): Sequential {
        this.loss = loss
        return this
    }

    fun build(): Sequential {
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
            if (layer !is Input) {
                grad = layer.backward(grad)
            }
        }
    }

    // Train step
    fun trainStep(inputs: NDArray<Double, *>, yTrue: D2Array<Double>) {
        outputShape = yTrue.shape
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
        epochs: Int
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

    // Predict function with batch processing
    fun predict(
        X: NDArray<Double, *>
    ): NDArray<Double, *> {
        val nSamples = X.shape[0]
        val predictions = mk.zeros<Double>(nSamples, outputShape[1])

        // Process the input in batches
        for (startIdx in 0 until nSamples step batchSize) {
            val endIdx = minOf(startIdx + batchSize, nSamples)
            val XBatch = X[startIdx until endIdx]

            // Perform forward pass on the batch
            val batchPredictions = forward(XBatch)
            for (batchIndex in 0 until endIdx - startIdx) {
                predictions[startIdx + batchIndex] = batchPredictions[batchIndex]
            }
        }

        // Combine all batch predictions into a single NDArray
        return predictions
    }
}

fun main() {
    val random = Random(42)
    // Create a model using the builder pattern
    val model = Sequential(batchSize = 3, random = random)
        .addInput(3) // Input layer with 3 features
        .addDense(30, ReLU()) // Hidden layer with 5 neurons and ReLU activation
        .addDense(3, Softmax()) // Output layer with 2 neurons and Softmax activation
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


    model.fit(inputs, yTrue, epochs = 100)
}