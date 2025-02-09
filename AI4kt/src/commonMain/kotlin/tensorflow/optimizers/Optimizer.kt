package tensorflow.optimizers

import tensorflow.layers.Conv2D
import tensorflow.layers.Dense
import tensorflow.layers.TrainableLayer

interface Optimizer {
    fun update(layer: TrainableLayer) {
        if (layer is Dense) {
            updateDence(layer)
        } else if (layer is Conv2D) {
            updateConv2D(layer)
        } else {
            throw IllegalArgumentException("Only DNNLayer & CNNLayer is supported")
        }
    }

    fun updateDence(layer: Dense) {
        throw Exception("not implemented")
    }

    fun updateConv2D(layer: Conv2D) {
        throw Exception("not implemented")
    }

    fun copy(): Optimizer
}