package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import static minicpbp.cp.Factory.*;
import java.util.Set;

public class ModeETR implements CPMode {

    private final int budget;

    public ModeETR() {
        this(0);
    }

    public ModeETR(int budget) {
        this.budget = budget;
    }

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward) {
        if (this.budget > 0) {
            int[] noSlipActions = {4, 5, 6, 7};
            cp.post(atmost(action, noSlipActions, this.budget));
            System.out.println("-> Contrainte BUDGET appliquée en mode ETR (Max " + this.budget + " mouvements sûrs).");
        }
        // En ETR, on explore toutes les possibilités (pas de contrainte sur totalReward)
    }

    @Override
    public int getNbActions() {
        return budget > 0 ? 8 : 4;
    }

    @Override
    public void fillTransitions(double[][][] P, int nbStates, int squareSize, Set<Integer> holeSet, int goalStateIdx, double noSlipProba, double sideSlipProba) {
        int nbA = getNbActions();
        for (int i = 0; i < nbStates; i++) {
            if (holeSet.contains(i) || i == goalStateIdx) {
                for (int j = 0; j < nbA; j++) {
                    P[i][j][i] = 1.0;
                }
            } else {
                // Actions 0-3 : stochastiques
                for (int j = 0; j < 4; j++) {
                    int s_intended, s_perp1, s_perp2;
                    switch (j) {
                        case 0: s_intended = GridNav.left(i, squareSize);            s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break;
                        case 1: s_intended = GridNav.below(i, squareSize, nbStates); s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break;
                        case 2: s_intended = GridNav.right(i, squareSize);           s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break;
                        case 3: s_intended = GridNav.above(i, squareSize);           s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break;
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

                // Actions 4-7 : déterministes (uniquement si budget > 0)
                if (budget > 0) {
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
        }
    }
}
