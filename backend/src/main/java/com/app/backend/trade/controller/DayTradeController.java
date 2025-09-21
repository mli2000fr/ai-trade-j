package com.app.backend.trade.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/min")
public class DayTradeController {


    @Autowired
    private DayTradeHelper dayTradeHelper;

    @GetMapping("/updateMinValue")
    public int updateMinValue(@RequestParam(value = "symbol", required = false) String symbol) throws InterruptedException {
        return dayTradeHelper.alimenteDBMinValue(symbol);
    }


}