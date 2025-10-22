package com.app.backend.nd4j;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

class DefaultDataTypeTest {

    @Test
    void defaultFloatingPointTypeIsFloat() {
        DataType fp = Nd4j.defaultFloatingPointType();
        assertEquals(DataType.FLOAT, fp, "Le type flottant ND4J par défaut devrait être FLOAT (32-bit)");
    }
}

