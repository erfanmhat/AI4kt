package io.ai4kt.ai4kt.fibonacci

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get


fun generateFibi() = sequence {
    var a = firstElement
    yield(a)
    var b = secondElement
    yield(b)
    while (a > 0 && b > 0) {
        val c = a + b
        yield(c)
        a = b
        b = c
    }
}

expect val firstElement: Int
expect val secondElement: Int

fun main() {
    val a = mk.ndarray(
        mk[
            mk[
                mk[1.5, 2.0, 3.0],
                mk[4.0, 5.0, 6.0]],
            mk[
                mk[3.0, 2.0, 1.0],
                mk[4.0, 5.0, 6.0]
            ]
        ]
    )
    println(a::class)
    println(a[1]::class)
//    println(a)
//    println(a)
}
