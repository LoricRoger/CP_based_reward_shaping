package minicpbp.examples;

/**
 * Classe utilitaire de test : mode ETR avec un budget de 3 (8 actions).
 * Utilisée uniquement dans les tests unitaires.
 */
class ModeBudget extends ModeETR {
    ModeBudget() {
        super(3);
    }
}
