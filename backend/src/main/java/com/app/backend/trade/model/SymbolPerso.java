package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class SymbolPerso {

    private String symbols;
    private String name;
    private String id;
    private String date;
}

