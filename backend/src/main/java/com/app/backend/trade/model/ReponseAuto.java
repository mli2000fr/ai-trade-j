package com.app.backend.trade.model;

import com.app.backend.trade.controller.TradeHelper;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Builder
@Setter
@Getter
@ToString
public class ReponseAuto {
    private Long idGpt;
    private List<OrderRequest> orders;
    private String analyseGpt;

}
