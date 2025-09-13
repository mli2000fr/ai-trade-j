package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

import java.sql.Date;

@Builder
@Data
public class SignalInfo {
    private String symbol;
    private SignalType type;
    private Date date;
    private String dateStr;
}

