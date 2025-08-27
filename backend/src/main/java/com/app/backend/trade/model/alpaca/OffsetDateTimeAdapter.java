package com.app.backend.trade.model.alpaca;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;

public class OffsetDateTimeAdapter extends TypeAdapter<OffsetDateTime> {
    @Override
    public void write(JsonWriter out, OffsetDateTime value) throws IOException {
        out.value(value != null ? value.toString() : null);
    }
    @Override
    public OffsetDateTime read(JsonReader in) throws IOException {
        if (in.peek() == com.google.gson.stream.JsonToken.NULL) {
            in.nextNull();
            return null;
        }
        String str = in.nextString();
        return str != null ? OffsetDateTime.parse(str, DateTimeFormatter.ISO_OFFSET_DATE_TIME) : null;
    }
}
