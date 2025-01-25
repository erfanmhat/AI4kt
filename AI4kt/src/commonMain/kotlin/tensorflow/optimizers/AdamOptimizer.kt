package io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers

import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.pow

class AdamOptimizer(
    val learningRate: Double = 0.001,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-8
) : Optimizer {

    // First moment estimates (mean of gradients)
    private var mWeights: D2Array<Double>? = null
    private var mBiases: D1Array<Double>? = null

    // Second moment estimates (uncentered variance of gradients)
    private var vWeights: D2Array<Double>? = null
    private var vBiases: D1Array<Double>? = null

    // Time step (for bias correction)
    private var t: Int = 0

    override fun update(layer: DNNLayer) {
        // Initialize moment estimates if they are null
        if (mWeights == null || vWeights == null) {
            // Initialize with the same shape as layer.weights
            mWeights = mk.zeros(layer.weights.shape[0], layer.weights.shape[1])
            vWeights = mk.zeros(layer.weights.shape[0], layer.weights.shape[1])
        }
        if (mBiases == null || vBiases == null) {
            // Initialize with the same shape as layer.biases
            mBiases = mk.zeros(layer.biases.size)
            vBiases = mk.zeros(layer.biases.size)
        }

        // Increment time step
        t += 1

        // Update first moment estimates (mean)
        mWeights = beta1 * mWeights!! + (1.0 - beta1) * layer.dweights
        mBiases = beta1 * mBiases!! + (1.0 - beta1) * layer.dbiases

        // Update second moment estimates (variance)
        vWeights = beta2 * vWeights!! + (1.0 - beta2) * (layer.dweights * layer.dweights)
        vBiases = beta2 * vBiases!! + (1.0 - beta2) * (layer.dbiases * layer.dbiases)

        // Bias-corrected first moment estimates
        val mWeightsHat = mWeights!! / (1.0 - beta1.pow(t))
        val mBiasesHat = mBiases!! / (1.0 - beta1.pow(t))

        // Bias-corrected second moment estimates
        val vWeightsHat = vWeights!! / (1.0 - beta2.pow(t))
        val vBiasesHat = vBiases!! / (1.0 - beta2.pow(t))

        // Update weights and biases
        layer.weights = layer.weights - learningRate * mWeightsHat / (vWeightsHat.map { it.pow(0.5) } + epsilon)
        layer.biases = layer.biases - learningRate * mBiasesHat / (vBiasesHat.map { it.pow(0.5) } + epsilon)
    }
}