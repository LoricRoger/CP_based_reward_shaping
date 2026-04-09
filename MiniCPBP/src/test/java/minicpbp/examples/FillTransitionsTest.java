package minicpbp.examples;

import org.junit.Test;
import java.util.HashSet;
import java.util.Set;
import static org.junit.Assert.*;

/**
 * Teste fillTransitions pour ModeMS, ModeETR et ModeBudget.
 *
 * Grille 4x4 standard FrozenLake :
 *  S  F  F  F
 *  F  H  F  H
 *  F  F  F  H
 *  H  F  F  G
 *
 * Actions : 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
 * Pour ModeBudget : 4-7 sont les versions déterministes de 0-3.
 */
public class FillTransitionsTest {

    private static final int SQUARE  = 4;
    private static final int N       = 16;
    private static final int GOAL    = 15;
    private static final double NO_SLIP  = 1.0 / 3.0;
    private static final double SIDE_SLIP = (1.0 - NO_SLIP) / 2.0;
    private static final double DELTA = 1e-9;

    private static Set<Integer> standardHoles() {
        Set<Integer> h = new HashSet<>();
        h.add(5); h.add(7); h.add(11); h.add(12);
        return h;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Vérifie que chaque ligne (état, action) somme à 1. */
    private void assertProbaSumToOne(double[][][] P, int nbStates, int nbActions) {
        for (int i = 0; i < nbStates; i++) {
            for (int j = 0; j < nbActions; j++) {
                double sum = 0;
                for (int k = 0; k < nbStates; k++) sum += P[i][j][k];
                assertEquals("P[" + i + "][" + j + "] ne somme pas à 1 (=" + sum + ")", 1.0, sum, DELTA);
            }
        }
    }

    /** Vérifie le self-loop sur les états terminaux. */
    private void assertTerminalSelfLoop(double[][][] P, Set<Integer> holes, int goal, int nbActions) {
        Set<Integer> terminals = new HashSet<>(holes);
        terminals.add(goal);
        for (int i : terminals) {
            for (int j = 0; j < nbActions; j++) {
                assertEquals("Self-loop manquant : P[" + i + "][" + j + "][" + i + "]", 1.0, P[i][j][i], DELTA);
                for (int k = 0; k < P[i][j].length; k++) {
                    if (k != i) assertEquals("P[" + i + "][" + j + "][" + k + "] devrait être 0", 0.0, P[i][j][k], DELTA);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // ModeMS
    // -------------------------------------------------------------------------

    @Test
    public void modeMS_probasSommentA1() {
        double[][][] P = new double[N][4][N];
        new ModeMS().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertProbaSumToOne(P, N, 4);
    }

    @Test
    public void modeMS_selfLoopTerminaux() {
        double[][][] P = new double[N][4][N];
        new ModeMS().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertTerminalSelfLoop(P, standardHoles(), GOAL, 4);
    }

    @Test
    public void modeMS_etatCentre_actionDroite_probas() {
        // État 5 est un trou — on teste l'état 6 (non-terminal, intérieur)
        // Action 2 = RIGHT : intended=7(trou), perp1=above(6)=2, perp2=below(6)=10
        double[][][] P = new double[N][4][N];
        new ModeMS().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertEquals(NO_SLIP,   P[6][2][7],  DELTA); // intended -> état 7 (trou, mais transition valide)
        assertEquals(SIDE_SLIP, P[6][2][2],  DELTA); // glissement haut
        assertEquals(SIDE_SLIP, P[6][2][10], DELTA); // glissement bas
    }

    @Test
    public void modeMS_coinHautGauche_blocage() {
        // État 0 : left et above sont bloqués (self-loop), donc s_perp peut être identique
        // Action 3 = UP : intended=above(0)=0, perp1=left(0)=0, perp2=right(0)=1
        // Quand intended==perp1==0 : P[0][3][0] += NO_SLIP + SIDE_SLIP, P[0][3][1] += SIDE_SLIP
        double[][][] P = new double[N][4][N];
        Set<Integer> noHoles = new HashSet<>();
        new ModeMS().fillTransitions(P, N, SQUARE, noHoles, GOAL, NO_SLIP, SIDE_SLIP);
        // La somme doit toujours valoir 1
        double sum = 0;
        for (int k = 0; k < N; k++) sum += P[0][3][k];
        assertEquals(1.0, sum, DELTA);
    }

    // -------------------------------------------------------------------------
    // ModeETR — même logique de transition que MS
    // -------------------------------------------------------------------------

    @Test
    public void modeETR_probasSommentA1() {
        double[][][] P = new double[N][4][N];
        new ModeETR().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertProbaSumToOne(P, N, 4);
    }

    @Test
    public void modeETR_selfLoopTerminaux() {
        double[][][] P = new double[N][4][N];
        new ModeETR().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertTerminalSelfLoop(P, standardHoles(), GOAL, 4);
    }

    @Test
    public void modeETR_identique_modeMS() {
        double[][][] P_ms  = new double[N][4][N];
        double[][][] P_etr = new double[N][4][N];
        Set<Integer> holes = standardHoles();
        new ModeMS().fillTransitions(P_ms,  N, SQUARE, holes, GOAL, NO_SLIP, SIDE_SLIP);
        new ModeETR().fillTransitions(P_etr, N, SQUARE, holes, GOAL, NO_SLIP, SIDE_SLIP);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < N; k++)
                    assertEquals("P_ms["+i+"]["+j+"]["+k+"] != P_etr", P_ms[i][j][k], P_etr[i][j][k], DELTA);
    }

    // -------------------------------------------------------------------------
    // ModeBudget
    // -------------------------------------------------------------------------

    @Test
    public void modeBudget_probasSommentA1() {
        double[][][] P = new double[N][8][N];
        new ModeBudget().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertProbaSumToOne(P, N, 8);
    }

    @Test
    public void modeBudget_selfLoopTerminaux() {
        double[][][] P = new double[N][8][N];
        new ModeBudget().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        assertTerminalSelfLoop(P, standardHoles(), GOAL, 8);
    }

    @Test
    public void modeBudget_actionsStochastiques_identiques_modeMS() {
        // Les actions 0-3 de Budget doivent être identiques à ModeMS
        double[][][] P_ms     = new double[N][4][N];
        double[][][] P_budget = new double[N][8][N];
        Set<Integer> holes = standardHoles();
        new ModeMS().fillTransitions(P_ms,     N, SQUARE, holes, GOAL, NO_SLIP, SIDE_SLIP);
        new ModeBudget().fillTransitions(P_budget, N, SQUARE, holes, GOAL, NO_SLIP, SIDE_SLIP);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < N; k++)
                    assertEquals("Budget action " + j + " != MS pour état " + i, P_ms[i][j][k], P_budget[i][j][k], DELTA);
    }

    @Test
    public void modeBudget_actionsDeterministes_proba1() {
        // Les actions 4-7 doivent avoir exactement une destination à proba 1.0
        double[][][] P = new double[N][8][N];
        new ModeBudget().fillTransitions(P, N, SQUARE, standardHoles(), GOAL, NO_SLIP, SIDE_SLIP);
        Set<Integer> terminals = new HashSet<>(standardHoles());
        terminals.add(GOAL);
        for (int i = 0; i < N; i++) {
            if (terminals.contains(i)) continue; // self-loop déjà testé
            for (int j = 4; j < 8; j++) {
                double maxProba = 0;
                int count1 = 0;
                for (int k = 0; k < N; k++) {
                    if (P[i][j][k] > DELTA) count1++;
                    if (P[i][j][k] > maxProba) maxProba = P[i][j][k];
                }
                assertEquals("Action déterministe " + j + " de l'état " + i + " doit avoir exactement 1 destination", 1, count1);
                assertEquals("Action déterministe " + j + " de l'état " + i + " doit avoir proba 1.0", 1.0, maxProba, DELTA);
            }
        }
    }

    @Test
    public void modeBudget_actionDeterministe_memeDestination_queStochastique() {
        // Action j+4 doit aller au même intended que action j (sans slip)
        // On vérifie que l'état destination de j+4 est le même que le max de j
        double[][][] P = new double[N][8][N];
        Set<Integer> noHoles = new HashSet<>(); // sans trous pour éviter terminaux
        new ModeBudget().fillTransitions(P, N, SQUARE, noHoles, GOAL, NO_SLIP, SIDE_SLIP);

        // État 5 (intérieur), action 2=RIGHT : intended = 6
        // Action 6 (RIGHT déterministe) doit aller en 6 avec proba 1
        assertEquals(1.0, P[5][6][6], DELTA);

        // État 5, action 1=DOWN : intended = 9
        // Action 5 (DOWN déterministe) doit aller en 9 avec proba 1
        assertEquals(1.0, P[5][5][9], DELTA);

        // État 5, action 0=LEFT : intended = 4
        // Action 4 (LEFT déterministe) doit aller en 4 avec proba 1
        assertEquals(1.0, P[5][4][4], DELTA);

        // État 5, action 3=UP : intended = 1
        // Action 7 (UP déterministe) doit aller en 1 avec proba 1
        assertEquals(1.0, P[5][7][1], DELTA);
    }
}
