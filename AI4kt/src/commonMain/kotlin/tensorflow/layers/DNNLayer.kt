package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.Activation
import io.ai4kt.ai4kt.fibonacci.tensorflow.plusD1Array
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random

class DNNLayer(n_inputs: Int, n_neurons: Int, private val activation: Activation? = null) {
    var weights: D2Array<Double>
    var biases: D1Array<Double>

    // Gradients for weights and biases
    var dweights: D2Array<Double> = mk.zeros<Double>(n_inputs, n_neurons)
    var dbiases: D1Array<Double> = mk.zeros<Double>(n_neurons)

    // Cache for inputs during forward pass (used in backward pass)
    private lateinit var inputs: D2Array<Double>

    init {
        weights = mk.ndarray(List(n_inputs) { List(n_neurons) { Random.nextDouble(-0.1, 0.1) } })
        biases = mk.zeros<Double>(n_neurons)
    }

    fun forward(inputs: D2Array<Double>): D2Array<Double> {
        // Cache the inputs for use in the backward pass
        this.inputs = inputs

        // Linear transformation
        val output = mk.linalg.dot(inputs, weights).plusD1Array(biases)

        // Apply activation function if provided
        return activation?.forward(output) ?: output
    }

    fun backward(dvalues: D2Array<Double>): D2Array<Double> {
        // Gradients for activation function
        val dactivation = activation?.backward(dvalues, inputs) ?: dvalues

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
    val l1 = DNNLayer(4, 5)
    val l2 = DNNLayer(5, 2)
    val l1_out = l1.forward(X)
    val l2_out = l2.forward(l1_out)
    println(l1_out)
    println()
    println(l2_out)
}