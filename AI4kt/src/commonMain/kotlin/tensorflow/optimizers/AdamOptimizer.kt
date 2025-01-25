package io.ai4kt.ai4kt.fibonacci.tensorflow.optimizers

import io.ai4kt.ai4kt.fibonacci.tensorflow.layers.DNNLayer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.pow
import kotlin.math.sqrt

class AdamOptimizer(
    private val learningRate: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-7
) : Optimizer {

    override fun copy(): AdamOptimizer {
        return AdamOptimizer(
            learningRate = learningRate,
            beta1 = beta1,
            beta2 = beta2,
            epsilon = epsilon
        )
    }

    private lateinit var m_weights: D2Array<Double>  // First moment estimates for weights
    private lateinit var v_weights: D2Array<Double>  // Second moment estimates for weights
    private lateinit var m_biases: D1Array<Double>   // First moment estimates for biases
    private lateinit var v_biases: D1Array<Double>   // Second moment estimates for biases
    private var t: Int = 0  // Timestep

    override fun update(layer: DNNLayer) {
        // Initialize moment estimates for weights and biases only once
        if (!::m_weights.isInitialized) {
            m_weights = mk.zeros(layer.weights.shape[0], layer.weights.shape[1])  // Initialize m for weights (D2Array)
            v_weights = mk.zeros(layer.weights.shape[0], layer.weights.shape[1])  // Initialize v for weights (D2Array)
            m_biases = mk.zeros(layer.biases.size)     // Initialize m for biases (D1Array)
            v_biases = mk.zeros(layer.biases.size)     // Initialize v for biases (D1Array)
        }

        t += 1  // Increment timestep

        // Gradients for the weights and biases
        val gradW = layer.dweights  // dweights is a D2Array
        val gradB = layer.dbiases   // dbiases is a D1Array

        // Update m and v for weights (D2Array)
        m_weights = beta1 * m_weights + (1 - beta1) * gradW
        v_weights = beta2 * v_weights + (1 - beta2) * gradW * gradW

        // Compute bias-corrected estimates for weights
        val m_hat_weights = m_weights / (1 - beta1.pow(t))
        val v_hat_weights = v_weights / (1 - beta2.pow(t))

        // Update the weights element-wise (row-wise or column-wise)
        for (i in 0 until layer.weights.shape[0]) {
            layer.weights[i] =
                layer.weights[i] - learningRate * (m_hat_weights[i] / (v_hat_weights[i].map { sqrt(it) } + epsilon))
        }

        // Update m and v for biases (D1Array)
        m_biases = beta1 * m_biases + (1 - beta1) * gradB
        v_biases = beta2 * v_biases + (1 - beta2) * gradB.map { it * it }

        // Compute bias-corrected estimates for biases
        val m_hat_biases = m_biases / (1 - beta1.pow(t))
        val v_hat_biases = v_biases / (1 - beta2.pow(t))

        // Update biases
        layer.biases = layer.biases - learningRate * (m_hat_biases / (v_hat_biases.map { sqrt(it) } + epsilon))
    }
}
