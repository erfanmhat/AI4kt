package tensorflow.optimizers

import tensorflow.layers.DNNLayer
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import tensorflow.layers.CNNLayer
import tensorflow.layers.TrainableLayer

class GradientDescentOptimizer(val learningRate: Double) : Optimizer {
    override fun updateDNN(layer: DNNLayer) {
        // Update weights
        layer.weights -= layer.dweights * learningRate

        // Update biases
        layer.biases -= layer.dbiases * learningRate
    }

    override fun updateCNN(layer: CNNLayer) {
        TODO()
    }

    override fun copy(): Optimizer {
        return GradientDescentOptimizer(learningRate)
    }
}