package tensorflow.optimizers

import tensorflow.layers.Conv2D
import tensorflow.layers.Dense
import tensorflow.layers.TrainableLayer

interface Optimizer {
    fun update(layer: TrainableLayer) {
        if (layer is Dense) {
            updateDNN(layer)
        } else if (layer is Conv2D) {
            updateCNN(layer)
        } else {
            throw IllegalArgumentException("Only DNNLayer & CNNLayer is supported")
        }
    }

    fun updateDNN(layer: Dense) {
        throw Exception("not implemented")
    }

    fun updateCNN(layer: Conv2D) {
        throw Exception("not implemented")
    }

    fun copy(): Optimizer
}