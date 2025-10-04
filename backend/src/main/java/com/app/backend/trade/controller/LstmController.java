package com.app.backend.trade.controller;

import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.PreditLsdm;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;

/**
 * Contrôleur REST exposant des endpoints pour :
 * 1. Entraîner un modèle LSTM sur un symbole (trainLstm)
 * 2. Prédire la prochaine valeur de clôture (predictNextClose)
 * 3. Lancer un tuning global sur tous les symboles (tuneAllSymbols)
 * 4. Consulter un reporting des exceptions rencontrées pendant le tuning (getTuningExceptionReport)
 *
 * Objectif pédagogique :
 * - Cette classe sert uniquement d'interface HTTP (elle délègue la logique métier au helper LstmHelper).
 * - Elle NE DOIT PAS contenir de logique d'apprentissage elle-même.
 *
 * Important :
 * - Ne pas ajouter de logique ici pour éviter toute régression.
 * - Toute modification fonctionnelle doit aller dans les services (ex : LstmHelper, LstmTuningService).
 *
 * Technologies utilisées :
 * - Spring REST (@RestController, @RequestMapping, @GetMapping)
 * - ND4J (backend numérique pour DeepLearning4J)
 * - SLF4J (journalisation)
 */
@RestController
@RequestMapping("/api/lstm") // Préfixe commun à tous les endpoints LSTM
public class LstmController {

    // Logger centralisé pour tracer les actions de ce contrôleur.
    // Utiliser logger.debug(...) pour des détails supplémentaires si besoin.
    private static final Logger logger = LoggerFactory.getLogger(LstmController.class);

    /**
     * Helper (service "façade") injecté par Spring.
     * Il encapsule :
     * - L'entraînement des modèles
     * - La prédiction
     * - Le tuning
     * - Le reporting des erreurs
     *
     * Important : Injection par field conservée (même si l'injection par constructeur est recommandée dans les nouveaux projets)
     * pour NE PAS introduire de refactor structurel ici (principe : zéro risque de régression).
     */
    @Autowired
    private LstmHelper lsdmHelper;

    /**
     * Endpoint : Entraîne un modèle LSTM pour un symbole donné.
     *
     * Usage :
     *   GET /api/lstm/train?symbol=BTCUSDT
     *
     * Pré-conditions :
     * - Le symbole doit exister dans vos données source.
     * - Le service LstmHelper doit gérer l'absence éventuelle de données (validation côté service).
     *
     * Post-condition :
     * - Lance un entraînement asynchrone ou synchrone selon l'implémentation interne de lsdmHelper.trainLstm().
     *
     * @param symbol Code du marché (ex : BTCUSDT, ETHUSDT, etc.)
     * @return Message simple de confirmation (texte brut).
     *
     * NOTE : Pas de gestion d'erreur spécifique ici pour ne pas altérer le flux existant.
     * Toute exception non contrôlée remontera (Spring la gère via son mécanisme d’exception).
     */
    @GetMapping("/train")
    public String trainLstm(@RequestParam String symbol) {
        // Délégation directe sans transformation (pas de logique ici volontairement).
        lsdmHelper.trainLstm(symbol);
        return "Entraînement LSTM terminé pour " + symbol;
    }

    /**
     * Endpoint : Prédit la prochaine valeur de clôture pour un symbole donné.
     *
     * Exemple d'appel :
     *   GET /api/lstm/predict?symbol=BTCUSDT
     *
     * @param symbol Symbole cible.
     * @return Objet PreditLsdm contenant la prédiction (structure définie dans le modèle).
     * @throws IOException si une ressource nécessaire (fichier modèle, données) est indisponible.
     *
     * Remarques :
     * - Aucune transformation ici : on renvoie directement la valeur fournie par le helper.
     * - Si vous devez enrichir la réponse (ex : ajouter un horodatage), faites-le dans le service ou créez un DTO dédié.
     */
    @GetMapping("/predict")
    public PreditLsdm predictNextClose(@RequestParam String symbol) throws IOException {
        return lsdmHelper.getPredit(symbol);
    }

    /**
     * Endpoint : Lance le tuning (ajustement d’hyperparamètres / exploration) sur tous les symboles configurés.
     *
     * Exemple :
     *   GET /api/lstm/tuneAllSymbols
     *
     * Étapes internes (commentaires ligne à ligne dans la méthode) :
     * - Log du backend ND4J utilisé (utile pour diagnostiquer CPU vs GPU).
     * - Configuration explicite des types numériques par défaut.
     * - Activation de l'accès cross-device (utile si plusieurs devices / GPU).
     * - Appel du helper pour effectuer le tuning.
     *
     * Retour :
     * - Toujours 'true' actuellement (indique simplement que la requête a été acceptée / terminée).
     *
     * Améliorations futures possibles (TODO) :
     * - Exposer un état d'avancement.
     * - Renvoyer un objet avec nombre de succès / erreurs.
     * - Exécuter de façon asynchrone (ex : @Async) pour ne pas bloquer le thread HTTP.
     */
    @GetMapping("/tuneAllSymbols")
    public boolean tuneAllSymbols() {
        // (Option décommentable) Forcer un backend CPU spécifique si problème de compatibilité :
        // System.setProperty("org.nd4j.linalg.defaultbackend", "org.nd4j.linalg.cpu.nativecpu.CpuBackend");

        // Log de départ : utile pour mesurer la durée (associer éventuellement à un chrono externe).
        logger.info("start------------------- -> " + Nd4j.getExecutioner().getClass().getSimpleName());

        // Affiche le backend ND4J (ex: CpuBackend, CudaBackend).
        logger.info("Backend: " + Nd4j.getExecutioner().getClass().getSimpleName());

        // Fournisseur BLAS (implémentation des opérations matricielles). Utile si performances suspectes.
        logger.info("Blas Vendor: " + Nd4j.factory().blas().getBlasVendor());

        // Fixe les types numériques par défaut (FLOAT pour données + gradients).
        // Important : garder cohérent avec le reste du pipeline d'entraînement pour éviter conversions implicites.
        Nd4j.setDefaultDataTypes(
                org.nd4j.linalg.api.buffer.DataType.FLOAT,
                org.nd4j.linalg.api.buffer.DataType.FLOAT
        );

        // Permet au backend d'accéder à plusieurs devices (ex: multi-GPU) si supporté.
        // Si non nécessaire, laisser tel quel n'a pas d'effet néfaste.
        Nd4j.getAffinityManager().allowCrossDeviceAccess(true);

        // Appel central : déclenche le processus complet de tuning (internalisé dans LstmHelper).
        lsdmHelper.tuneAllSymbols();

        // Log de fin.
        logger.info("end------------------- -> " + Nd4j.getExecutioner().getClass().getSimpleName());

        // Retour simple (aucune sémantique avancée ici).
        return true;
    }

    /**
     * Endpoint : Récupère la liste des erreurs rencontrées lors des phases de tuning.
     *
     * Usage :
     *   GET /api/lstm/tuning-exceptions
     *
     * @return Liste d'entrées (symboles + messages d'erreur) fournie par LstmTuningService via le helper.
     *
     * Bonnes pratiques :
     * - Ce reporting peut être utilisé par un outil de supervision.
     * - Envisager pagination si la liste devient volumineuse.
     */
    @GetMapping("/tuning-exceptions")
    @ResponseBody
    public List<LstmTuningService.TuningExceptionReportEntry> getTuningExceptionReport() {
        return lsdmHelper.getTuningExceptionReport();
    }

    @GetMapping("/test-feature-matrix-cache")
    public String testFeatureMatrixCacheApi() {
        // Capture la sortie standard
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        try {
            System.setOut(new PrintStream(baos));
            LstmTradePredictor.testFeatureMatrixCache();
        } finally {
            System.setOut(oldOut);
        }
        return baos.toString();
    }
}
