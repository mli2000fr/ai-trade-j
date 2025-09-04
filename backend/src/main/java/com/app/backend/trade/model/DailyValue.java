package com.app.backend.trade.model;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
public class DailyValue {
    private String date;
    private String open;
    private String high;
    private String low;
    private String close;
    private String volume;
    private String numberOfTrades;
    private String volumeWeightedAveragePrice;
}

/*
 * c : close — prix de clôture de la bougie (ici 182.775)
 * h : high — prix le plus haut atteint pendant la période (185.86)
 * l : low — prix le plus bas atteint pendant la période (181.845)
 * n : number of trades — nombre de transactions durant la période (865)
 * o : open — prix d’ouverture de la bougie (185.86)
 * t : timestamp — date et heure de la bougie (2025-03-18T04:00:00Z)
 * v : volume — volume total échangé durant la période (22150)
 * vw : volume weighted average price — prix moyen pondéré par le volume (182.894341)
 */