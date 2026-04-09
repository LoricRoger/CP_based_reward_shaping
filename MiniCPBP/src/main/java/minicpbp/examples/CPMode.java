// CPMode.java
package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;

public interface CPMode {
    /**
     * Applique les contraintes spécifiques au mode sur le solveur.
     */
    void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget);
    int requiredNbActions();
}