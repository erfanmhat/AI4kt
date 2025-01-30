package tensorflow.optimizers

import tensorflow.layers.Dense
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import tensorflow.layers.Conv2D

class GradientDescentOptimizer(val learningRate: Double) : Optimizer {
    override fun updateDNN(layer: Dense) {
        // Update weights
        layer.weights -= layer.dweights * learningRate

        // Update biases
        layer.biases -= layer.dbiases * learningRate
    }

    override fun updateCNN(layer: Conv2D) {
        TODO()
    }

    override fun copy(): Optimizer {
        return GradientDescentOptimizer(learningRate)
    }
}