package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CompteDto {
    private Integer id;
    private String nom;
    private String alias;
    private Boolean real;
}
