package io.ai4kt.ai4kt.fibonacci

import kotlin.test.Test
import kotlin.test.assertEquals

class AndroidFibiTest {

    @Test
    fun `test 3rd element`() {
        assertEquals(3, generateFibi().take(3).last())
    }
}