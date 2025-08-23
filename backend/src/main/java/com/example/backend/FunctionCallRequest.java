package com.example.backend;

import java.util.Map;

public class FunctionCallRequest {
    private String functionName;
    private Map<String, String> arguments;

    public String getFunctionName() {
        return functionName;
    }

    public void setFunctionName(String functionName) {
        this.functionName = functionName;
    }

    public Map<String, String> getArguments() {
        return arguments;
    }

    public void setArguments(Map<String, String> arguments) {
        this.arguments = arguments;
    }
}

