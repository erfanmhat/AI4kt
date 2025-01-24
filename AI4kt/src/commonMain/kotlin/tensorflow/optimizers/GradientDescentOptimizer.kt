package io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers

import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class GradientDescentOptimizer(val learningRate: Double) {
    fun update(layer: DNNLayer) {
        // Update weights
        layer.weights -= layer.dweights * learningRate

        // Update biases
        layer.biases -= layer.dbiases * learningRate
    }
}