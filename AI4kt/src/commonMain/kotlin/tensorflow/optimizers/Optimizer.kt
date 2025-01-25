package io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers

import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer

interface Optimizer {
    fun update(layer: DNNLayer)
    fun copy(): Optimizer
}