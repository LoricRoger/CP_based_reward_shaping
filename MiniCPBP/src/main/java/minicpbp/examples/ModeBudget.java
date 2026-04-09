package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import static minicpbp.cp.Factory.*;
import java.util.Set;

public class ModeBudget implements CPMode {

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget) {
        // 1. Contrainte MS classique
        if (goalReward > 0) {
            totalReward.removeBelow(goalReward);
        }

        // 2. Contrainte de budget sur les actions "NoSlip" (valeurs 4, 5, 6, 7)
        int[] noSlipActions = {4, 5, 6, 7};
        cp.post(atmost(action, noSlipActions, budget));

        System.out.println("-> Contraintes BUDGET appliquées (Max " + budget + " mouvements sûrs).");
    }

    @Override
    public int getNbActions() {
        return 8;
    }

    @Override
    public void fillTransitions(double[][][] P, int nbStates, int squareSize, Set<Integer> holeSet, int goalStateIdx, double noSlipProba, double sideSlipProba) {
        for (int i = 0; i < nbStates; i++) {
            if (holeSet.contains(i) || i == goalStateIdx) {
                for (int j = 0; j < 8; j++) P[i][j][i] = 1.0;
            } else {
                // Actions 0-3 : transitions stochastiques (avec slip)
                for (int j = 0; j < 4; j++) {
                    int s_intended, s_perp1, s_perp2;
                    switch (j) {
                        case 0: s_intended = GridNav.left(i, squareSize);            s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break; // LEFT
                        case 1: s_intended = GridNav.below(i, squareSize, nbStates); s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break; // DOWN
                        case 2: s_intended = GridNav.right(i, squareSize);           s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break; // RIGHT
                        case 3: s_intended = GridNav.above(i, squareSize);           s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break; // UP
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
                    if (sum_k > 1e-9 && Math.abs(sum_k - 1.0) > 1e-9) {
                        for (int k = 0; k < nbStates; k++) P[i][j][k] /= sum_k;
                    } else if (sum_k <= 1e-9) {
                        P[i][j][i] = 1.0;
                    }
                }

                // Actions 4-7 : versions déterministes des actions 0-3 (même destination, proba 1.0)
                for (int j = 0; j < 4; j++) {
                    int s_intended;
                    switch (j) {
                        case 0: s_intended = GridNav.left(i, squareSize);            break; // LEFT  déterministe
                        case 1: s_intended = GridNav.below(i, squareSize, nbStates); break; // DOWN  déterministe
                        case 2: s_intended = GridNav.right(i, squareSize);           break; // RIGHT déterministe
                        case 3: s_intended = GridNav.above(i, squareSize);           break; // UP    déterministe
                        default: s_intended = i;
                    }
                    P[i][j + 4][s_intended] = 1.0;
                }
            }
        }
    }
}
