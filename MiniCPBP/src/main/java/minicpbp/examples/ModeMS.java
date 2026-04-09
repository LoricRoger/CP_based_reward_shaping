package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import java.util.Set;

public class ModeMS implements CPMode {

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget) {
        if (budget > 0) {
            System.err.println("WARN: ModeMS reçoit un budget=" + budget + " mais ne l'utilise pas.");
        }
        if (goalReward > 0) {
            totalReward.removeBelow(goalReward);
        } else if (goalReward == 0 && holeReward == 0) {
            totalReward.assign(0);
        }
    }

    @Override
    public int getNbActions() {
        return 4;
    }

    @Override
    public void fillTransitions(double[][][] P, int nbStates, int squareSize, Set<Integer> holeSet, int goalStateIdx, double noSlipProba, double sideSlipProba) {
        for (int i = 0; i < nbStates; i++) {
            if (holeSet.contains(i) || i == goalStateIdx) {
                for (int j = 0; j < 4; j++) {
                    P[i][j][i] = 1.0;
                }
            } else {
                for (int j = 0; j < 4; j++) {
                    int s_intended, s_perp1, s_perp2;
                    switch (j) {
                        case 0: s_intended = GridNav.left(i, squareSize);         s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break; // LEFT
                        case 1: s_intended = GridNav.below(i, squareSize, nbStates); s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break; // DOWN
                        case 2: s_intended = GridNav.right(i, squareSize);        s_perp1 = GridNav.above(i, squareSize); s_perp2 = GridNav.below(i, squareSize, nbStates); break; // RIGHT
                        case 3: s_intended = GridNav.above(i, squareSize);        s_perp1 = GridNav.left(i, squareSize);  s_perp2 = GridNav.right(i, squareSize);           break; // UP
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
            }
        }
    }
}
