package com.app.backend.trade.lstm;

/**
 * Cette classe regroupe tous les hyperparamètres nécessaires à l'entraînement
 * d'un modèle LSTM (Long Short-Term Memory) pour des tâches de prédiction de séries temporelles.
 *
 * IMPORTANT:
 * - Les attributs sont publics: ne pas changer leur visibilité sans vérifier tout le code existant.
 * - Aucun traitement ici: c'est un simple conteneur de données (Data Holder / DTO).
 * - Toute modification de nom de champ ou de type peut provoquer des régressions (ex: sérialisation JSON, accès direct).
 *
 * UTILISATION TYPIQUE:
 *  1. Créer une instance (via le constructeur vide ou complet).
 *  2. Lire/écrire les valeurs directement (ex: obj.windowSize = 60;).
 *  3. Transmettre l'objet à un service d'entraînement.
 *
 * Améliorations ajoutées:
 *  - Documentation détaillée de chaque paramètre.
 *  - Clarification des rôles pour un mainteneur débutant.
 */
public class LstmHyperparameters {

    /**
     * Taille de la fenêtre (nombre de pas de temps historiques utilisés comme entrée d'une séquence).
     * Exemple: si windowSize = 60, on utilise 60 points passés pour prédire le suivant.
     * Impact: plus grand => capture mieux le contexte, mais augmente le coût mémoire/temps.
     */
    public int windowSize;

    /**
     * Nombre de neurones (unités) dans la couche LSTM principale.
     * Plus élevé => capacité d'apprentissage accrue mais risque de surapprentissage (overfitting).
     */
    public int lstmNeurons;

    /**
     * Taux de dropout (entre 0.0 et 1.0). Exemple: 0.2 = 20% des neurones ignorés aléatoirement pendant l'entraînement.
     * Objectif: réduire le surapprentissage.
     */
    public double dropoutRate;

    /**
     * Taux d'apprentissage (learning rate) utilisé par l'optimiseur.
     * Valeur trop grande => instable. Trop petite => apprentissage très lent.
     */
    public double learningRate;

    /**
     * Nombre maximal d'époques (boucles complètes sur les données d'entraînement).
     * Peut être interrompu plus tôt si une stratégie d'early stopping est utilisée.
     */
    public int numEpochs;

    /**
     * Paramètre d'early stopping: nombre d'époques consécutives sans amélioration suffisante avant arrêt.
     */
    public int patience;

    /**
     * Amélioration minimale (delta) requise sur la métrique surveillée pour être considérée comme un progrès.
     * Utilisé avec l'early stopping (ex: minDelta = 0.0001).
     */
    public double minDelta;

    /**
     * Nom de l'optimiseur (ex: "adam", "rmsprop", "sgd").
     * Doit correspondre au moteur d'entraînement utilisé ailleurs.
     */
    public String optimizer;

    /**
     * Coefficient de régularisation L1 (pénalise la somme des valeurs absolues des poids).
     * Utiliser 0.0 si non souhaité.
     */
    public double l1;

    /**
     * Coefficient de régularisation L2 (pénalise la somme des carrés des poids).
     * Aide à réduire le surapprentissage.
     */
    public double l2;

    /**
     * Portée (scope) de la normalisation des données.
     * Exemple possible: "GLOBAL", "WINDOW", "FEATURE". Dépend de l'implémentation externe.
     */
    public String normalizationScope;

    /**
     * Méthode de normalisation (ex: "STANDARD", "MIN_MAX", "ROBUST").
     * Doit être cohérente avec le prétraitement appliqué ailleurs.
     */
    public String normalizationMethod;

    /**
     * Constructeur vide obligatoire pour:
     *  - Frameworks (Jackson, Hibernate, etc.)
     *  - Création suivie d'assignations manuelles.
     *
     * Ne pas supprimer.
     */
    public LstmHyperparameters() {}

    /**
     * Constructeur complet: permet d'initialiser tous les hyperparamètres en une seule étape.
     * L'ordre des paramètres ne doit pas être modifié (sinon risque de confusion / régressions).
     *
     * @param windowSize voir windowSize
     * @param lstmNeurons voir lstmNeurons
     * @param dropoutRate voir dropoutRate
     * @param learningRate voir learningRate
     * @param numEpochs voir numEpochs
     * @param patience voir patience
     * @param minDelta voir minDelta
     * @param optimizer voir optimizer
     * @param l1 voir l1
     * @param l2 voir l2
     * @param normalizationScope voir normalizationScope
     * @param normalizationMethod voir normalizationMethod
     */
    public LstmHyperparameters(int windowSize,
                               int lstmNeurons,
                               double dropoutRate,
                               double learningRate,
                               int numEpochs,
                               int patience,
                               double minDelta,
                               String optimizer,
                               double l1,
                               double l2,
                               String normalizationScope,
                               String normalizationMethod) {
        // Affectations directes: NE PAS MODIFIER (ordre cohérent avec les paramètres)
        this.windowSize = windowSize;
        this.lstmNeurons = lstmNeurons;
        this.dropoutRate = dropoutRate;
        this.learningRate = learningRate;
        this.numEpochs = numEpochs;
        this.patience = patience;
        this.minDelta = minDelta;
        this.optimizer = optimizer;
        this.l1 = l1;
        this.l2 = l2;
        this.normalizationScope = normalizationScope;
        this.normalizationMethod = normalizationMethod;
    }

    /**
     * Fournit une représentation lisible de l'objet.
     * Utile pour le logging ou le debug.
     * Ajout sans impact sur la logique métier.
     */
    @Override
    public String toString() {
        return "LstmHyperparameters{" +
                "windowSize=" + windowSize +
                ", lstmNeurons=" + lstmNeurons +
                ", dropoutRate=" + dropoutRate +
                ", learningRate=" + learningRate +
                ", numEpochs=" + numEpochs +
                ", patience=" + patience +
                ", minDelta=" + minDelta +
                ", optimizer='" + optimizer + '\'' +
                ", l1=" + l1 +
                ", l2=" + l2 +
                ", normalizationScope='" + normalizationScope + '\'' +
                ", normalizationMethod='" + normalizationMethod + '\'' +
                '}';
    }
}
