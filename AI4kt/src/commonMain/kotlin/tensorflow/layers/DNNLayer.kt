package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Activation
import io.ai4kt.ai4kt.fibonacci.tensorflow.plusD1Array
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import kotlin.math.sqrt
import kotlin.random.Random

class DNNLayer(
    n_inputs: Int,
    n_neurons: Int,
    private val random: Random,
    private val activation: Activation? = null
) : Layer {
    var weights: D2Array<Double>
    var biases: D1Array<Double>

    // Gradients for weights and biases
    var dweights: D2Array<Double> = mk.zeros<Double>(n_inputs, n_neurons)
    var dbiases: D1Array<Double> = mk.zeros<Double>(n_neurons)

    // Cache for inputs during forward pass (used in backward pass)
    private lateinit var inputs: D2Array<Double>

    init {
        // Xavier/Glorot initialization for weights
        val scale = sqrt(2.0 / (n_inputs + n_neurons))
        weights = mk.ndarray(List(n_inputs) { List(n_neurons) { random.nextDouble(-scale, scale) } })

        // Initialize biases to a small positive value
        biases = mk.zeros<Double>(n_neurons).map { 0.01 }
    }

    override fun forward(inputs: D2Array<Double>): D2Array<Double> {
        // Cache the inputs for use in the backward pass
        this.inputs = inputs

        // Linear transformation
        val output = mk.linalg.dot(inputs, weights).plusD1Array(biases)
//        val output = mk.linalg.dot(inputs, weights) + biases.broadcast(biases.size)todo


        // Apply activation function if provided
        return activation?.forward(output) ?: output
    }

    override fun backward(dvalues: D2Array<Double>): D2Array<Double> {
        // Gradients for activation function (if any)
        val dactivation = if (activation != null) {
            // Pass the output of the layer (before activation) to the activation's backward method
            val output = mk.linalg.dot(inputs, weights).plusD1Array(biases)
            activation.backward(dvalues, output)
        } else {
            dvalues
        }

        // Gradients for weights
        dweights = mk.linalg.dot(inputs.transpose(), dactivation)

        // Gradients for biases (sum over the batch axis)
        dbiases = mk.math.sum(dactivation, axis = 0)

        // Gradients for inputs
        val dinputs = mk.linalg.dot(dactivation, weights.transpose())

        return dinputs
    }
}

fun main() {
    val X = mk.ndarray(
        mk[
            mk[1.0, 2.0, 3.0, 2.5],
            mk[2.0, 5.0, -1.0, 2.0],
            mk[-1.5, 2.7, 3.3, -0.8]
        ]
    )
    val random = Random(42)
    val l1 = DNNLayer(4, 5, random)
    val l2 = DNNLayer(5, 2, random)
    val l1_out = l1.forward(X)
    val l2_out = l2.forward(l1_out)
    println(l1_out)
    println()
    println(l2_out)
}