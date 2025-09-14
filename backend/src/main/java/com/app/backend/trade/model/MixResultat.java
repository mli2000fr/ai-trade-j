package com.app.backend.trade.model;

import com.app.backend.trade.strategy.BestInOutStrategy;
import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class MixResultat {

    BestInOutStrategy single;
    BestCombinationResult mix;
}
