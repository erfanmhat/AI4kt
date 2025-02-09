package tensorflow.optimizers

import tensorflow.layers.Dense
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.layers.Conv2D
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

    private lateinit var m_weights: NDArray<Double, *>  // First moment estimates for weights
    private lateinit var v_weights: NDArray<Double, *>  // Second moment estimates for weights
    private lateinit var m_biases: D1Array<Double>   // First moment estimates for biases
    private lateinit var v_biases: D1Array<Double>   // Second moment estimates for biases
    private var t: Int = 0  // Timestep

    override fun updateDence(layer: Dense) {
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
        m_weights = beta1 * (m_weights as D2Array<Double>) + (1 - beta1) * gradW
        v_weights = beta2 * (v_weights as D2Array<Double>) + (1 - beta2) * gradW * gradW

        // Compute bias-corrected estimates for weights
        val m_hat_weights = (m_weights as D2Array<Double>) / (1 - beta1.pow(t))
        val v_hat_weights = (v_weights as D2Array<Double>) / (1 - beta2.pow(t))

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

    override fun updateConv2D(layer: Conv2D) {
        // Initialize moment estimates for weights and biases only once
        if (!::m_weights.isInitialized) {
            // Initialize m and v for weights (4D tensor: [kernelHeight, kernelWidth, inputChannels, filters])
            m_weights =
                mk.zeros(layer.weights.shape[0], layer.weights.shape[1], layer.weights.shape[2], layer.weights.shape[3])
            v_weights =
                mk.zeros(layer.weights.shape[0], layer.weights.shape[1], layer.weights.shape[2], layer.weights.shape[3])

            // Initialize m and v for biases (1D tensor: [filters])
            m_biases = mk.zeros(layer.biases.size)
            v_biases = mk.zeros(layer.biases.size)
        }

        t += 1  // Increment timestep

        // Gradients for the weights and biases
        val gradW = layer.dweights  // dweights is a 4D tensor: [kernelHeight, kernelWidth, inputChannels, filters]
        val gradB = layer.dbiases   // dbiases is a 1D tensor: [filters]

        // Update m and v for weights (4D tensor)
        m_weights = beta1 * (m_weights as D4Array<Double>) + (1 - beta1) * gradW
        v_weights = beta2 * (v_weights as D4Array<Double>) + (1 - beta2) * gradW * gradW

        // Compute bias-corrected estimates for weights
        val m_hat_weights = m_weights / (1 - beta1.pow(t))
        val v_hat_weights = v_weights / (1 - beta2.pow(t))

        // Update the weights element-wise
        for (kh in 0 until layer.weights.shape[0]) {
            for (kw in 0 until layer.weights.shape[1]) {
                for (ic in 0 until layer.weights.shape[2]) {
                    for (oc in 0 until layer.weights.shape[3]) {
                        layer.weights[kh, kw, ic, oc] = layer.weights[kh, kw, ic, oc] - learningRate *
                                (m_hat_weights[kh, kw, ic, oc] / (sqrt(v_hat_weights[kh, kw, ic, oc]) + epsilon))
                    }
                }
            }
        }

        // Update m and v for biases (1D tensor)
        m_biases = beta1 * m_biases + (1 - beta1) * gradB
        v_biases = beta2 * v_biases + (1 - beta2) * gradB.map { it * it }

        // Compute bias-corrected estimates for biases
        val m_hat_biases = m_biases / (1 - beta1.pow(t))
        val v_hat_biases = v_biases / (1 - beta2.pow(t))

        // Update biases
        layer.biases = layer.biases - learningRate * (m_hat_biases / (v_hat_biases.map { sqrt(it) } + epsilon))
    }
}
