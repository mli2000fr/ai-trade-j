package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ExecuteOrdersRequest {
    private String id;
    private List<OrderRequest> orders;
    private String idGpt;
}
