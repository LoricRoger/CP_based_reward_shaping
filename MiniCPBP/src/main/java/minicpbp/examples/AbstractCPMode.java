package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import static minicpbp.cp.Factory.*;
import java.util.Set;

/**
 * Base class for CP modes.
 *
 * Provides:
 *   - shared budget field and getNbActions() logic
 *   - shared fillTransitions() (stochastic actions 0-3 + deterministic actions 4-7 if budget > 0)
 *   - applyBudgetConstraint() helper that subclasses call inside applyConstraints()
 *
 * To add a new mode: extend this class and override applyConstraints() only.
 */
public abstract class AbstractCPMode implements CPMode {

    protected final int budget;

    protected AbstractCPMode(int budget) {
        this.budget = budget;
    }

    // -------------------------------------------------------------------------
    // CPMode interface — shared implementations
    // -------------------------------------------------------------------------

    @Override
    public int getNbActions() {
        return budget > 0 ? 8 : 4;
    }

    /**
     * Fills the transition probability matrix P[state][action][nextState].
     *
     * Actions 0-3 : stochastic (slippery), using noSlipProba / sideSlipProba.
     * Actions 4-7 : deterministic no-slip counterparts (only filled when budget > 0).
     *
     * Terminal states (holes and goal) loop on themselves for all actions.
     */
    @Override
    public void fillTransitions(double[][][] P, int nbStates, int squareSize,
                                Set<Integer> holeSet, int goalStateIdx,
                                double noSlipProba, double sideSlipProba) {
        int nbA = getNbActions();
        for (int i = 0; i < nbStates; i++) {
            if (holeSet.contains(i) || i == goalStateIdx) {
                // Terminal state: self-loop for all actions
                for (int j = 0; j < nbA; j++) {
                    P[i][j][i] = 1.0;
                }
            } else {
                fillStochasticTransitions(P, i, squareSize, nbStates, noSlipProba, sideSlipProba);
                if (budget > 0) {
                    fillDeterministicTransitions(P, i, squareSize, nbStates);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Protected helpers
    // -------------------------------------------------------------------------

    /**
     * Posts the atmost constraint limiting no-slip actions (4-7) to at most budget uses.
     * Call this from applyConstraints() when the subclass wants budget enforcement.
     */
    protected void applyBudgetConstraint(Solver cp, IntVar[] action) {
        if (budget > 0) {
            int[] noSlipActions = {4, 5, 6, 7};
            cp.post(atmost(action, noSlipActions, this.budget));
            System.out.println("-> Contrainte BUDGET appliquée (Max " + budget + " mouvements sûrs).");
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Fills stochastic transitions (actions 0-3) for a single non-terminal state. */
    private void fillStochasticTransitions(double[][][] P, int i, int squareSize, int nbStates,
                                           double noSlipProba, double sideSlipProba) {
        for (int j = 0; j < 4; j++) {
            int s_intended, s_perp1, s_perp2;
            switch (j) {
                case 0: s_intended = GridNav.left(i, squareSize);            s_perp1 = GridNav.above(i, squareSize);          s_perp2 = GridNav.below(i, squareSize, nbStates); break;
                case 1: s_intended = GridNav.below(i, squareSize, nbStates); s_perp1 = GridNav.left(i, squareSize);           s_perp2 = GridNav.right(i, squareSize);           break;
                case 2: s_intended = GridNav.right(i, squareSize);           s_perp1 = GridNav.above(i, squareSize);          s_perp2 = GridNav.below(i, squareSize, nbStates); break;
                case 3: s_intended = GridNav.above(i, squareSize);           s_perp1 = GridNav.left(i, squareSize);           s_perp2 = GridNav.right(i, squareSize);           break;
                default: s_intended = i; s_perp1 = i; s_perp2 = i;
            }

            if (s_perp1 == s_perp2) {
                P[i][j][s_intended] += noSlipProba;
                P[i][j][s_perp1]    += 2 * sideSlipProba;
            } else {
                P[i][j][s_intended] += noSlipProba;
                P[i][j][s_perp1]    += sideSlipProba;
                P[i][j][s_perp2]    += sideSlipProba;
            }

            // Normalize to guard against floating-point drift
            double sum_k = 0;
            for (int k = 0; k < nbStates; k++) sum_k += P[i][j][k];
            if (sum_k > 1e-9) {
                if (Math.abs(sum_k - 1.0) > 1e-9) {
                    for (int k = 0; k < nbStates; k++) P[i][j][k] /= sum_k;
                }
            } else {
                System.err.println("WARN: Zero probability sum for P[" + i + "][" + j + "]. Forcing self-loop.");
                P[i][j][i] = 1.0;
            }
        }
    }

    /** Fills deterministic transitions (actions 4-7) for a single non-terminal state. */
    private void fillDeterministicTransitions(double[][][] P, int i, int squareSize, int nbStates) {
        for (int j = 0; j < 4; j++) {
            int s_intended;
            switch (j) {
                case 0: s_intended = GridNav.left(i, squareSize);            break;
                case 1: s_intended = GridNav.below(i, squareSize, nbStates); break;
                case 2: s_intended = GridNav.right(i, squareSize);           break;
                case 3: s_intended = GridNav.above(i, squareSize);           break;
                default: s_intended = i;
            }
            P[i][j + 4][s_intended] = 1.0;
        }
    }
}
