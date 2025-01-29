package tensorflow.optimizers

import tensorflow.layers.CNNLayer
import tensorflow.layers.DNNLayer
import tensorflow.layers.TrainableLayer

interface Optimizer {
    fun update(layer: TrainableLayer) {
        if (layer is DNNLayer) {
            updateDNN(layer)
        } else if (layer is CNNLayer) {
            updateCNN(layer)
        } else {
            throw IllegalArgumentException("Only DNNLayer & CNNLayer is supported")
        }
    }

    fun updateDNN(layer: DNNLayer) {
        throw Exception("not implemented")
    }

    fun updateCNN(layer: CNNLayer) {
        throw Exception("not implemented")
    }

    fun copy(): Optimizer
}