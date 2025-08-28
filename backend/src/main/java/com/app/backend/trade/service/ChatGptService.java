package com.app.backend.trade.service;

import com.app.backend.trade.model.ChatGptResponse;
import com.app.backend.trade.util.TradeUtils;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.*;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

@Service
public class ChatGptService {
    @Value("${openai.api.key}")
    private String apiKey;

    @Value("${openai.api.model}")
    private String apiModel;


    public ChatGptResponse askChatGpt(String prompt) {

        // Création du client avec ta clé API
        OpenAIClient client = OpenAIOkHttpClient.builder().apiKey(apiKey).build();
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(apiModel)
                .addUserMessage(prompt)
                .build();
        try{
            TradeUtils.log("askChatGpt Prompt JSON : " + prompt);
            ChatCompletion response = client.chat().completions().create(params);
            String res = response.choices().get(0).message().content().orElse("aucune réponse donnée");
            TradeUtils.log("askChatGpt Réponse ChatGPT: " + res);
            return new ChatGptResponse(res, null);
        } catch (RestClientException e) {
            return new ChatGptResponse(null, "Erreur d'appel à l'API OpenAI : " + e.getMessage());
        }
    }

}
