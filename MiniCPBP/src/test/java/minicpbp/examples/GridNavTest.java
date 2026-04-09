package minicpbp.examples;

import org.junit.Test;
import static org.junit.Assert.*;

public class GridNavTest {

    // Grille 4x4 :
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15

    private static final int S = 4;
    private static final int N = 16;

    // --- left ---

    @Test
    public void left_milieu() {
        assertEquals(5, GridNav.left(6, S));
    }

    @Test
    public void left_bordGauche_selfLoop() {
        assertEquals(0, GridNav.left(0, S));
        assertEquals(4, GridNav.left(4, S));
        assertEquals(8, GridNav.left(8, S));
        assertEquals(12, GridNav.left(12, S));
    }

    @Test
    public void left_bordDroit() {
        assertEquals(2, GridNav.left(3, S));
        assertEquals(14, GridNav.left(15, S));
    }

    // --- right ---

    @Test
    public void right_milieu() {
        assertEquals(7, GridNav.right(6, S));
    }

    @Test
    public void right_bordDroit_selfLoop() {
        assertEquals(3,  GridNav.right(3,  S));
        assertEquals(7,  GridNav.right(7,  S));
        assertEquals(11, GridNav.right(11, S));
        assertEquals(15, GridNav.right(15, S));
    }

    @Test
    public void right_bordGauche() {
        assertEquals(1, GridNav.right(0, S));
        assertEquals(13, GridNav.right(12, S));
    }

    // --- above ---

    @Test
    public void above_milieu() {
        assertEquals(5, GridNav.above(9, S));
    }

    @Test
    public void above_ligneHaute_selfLoop() {
        assertEquals(0, GridNav.above(0, S));
        assertEquals(1, GridNav.above(1, S));
        assertEquals(2, GridNav.above(2, S));
        assertEquals(3, GridNav.above(3, S));
    }

    @Test
    public void above_ligneBasse() {
        assertEquals(8, GridNav.above(12, S));
        assertEquals(11, GridNav.above(15, S));
    }

    // --- below ---

    @Test
    public void below_milieu() {
        assertEquals(9, GridNav.below(5, S, N));
    }

    @Test
    public void below_ligneBasse_selfLoop() {
        assertEquals(12, GridNav.below(12, S, N));
        assertEquals(13, GridNav.below(13, S, N));
        assertEquals(14, GridNav.below(14, S, N));
        assertEquals(15, GridNav.below(15, S, N));
    }

    @Test
    public void below_ligneHaute() {
        assertEquals(4, GridNav.below(0, S, N));
        assertEquals(7, GridNav.below(3, S, N));
    }

    // --- coin haut-gauche (0) : cas limite multi-direction ---

    @Test
    public void coin_hautGauche() {
        assertEquals(0, GridNav.left(0,  S));     // bloqué à gauche
        assertEquals(0, GridNav.above(0, S));     // bloqué en haut
        assertEquals(1, GridNav.right(0, S));
        assertEquals(4, GridNav.below(0, S, N));
    }

    // --- coin bas-droit (15) ---

    @Test
    public void coin_basDroit() {
        assertEquals(15, GridNav.right(15, S));   // bloqué à droite
        assertEquals(15, GridNav.below(15, S, N));// bloqué en bas
        assertEquals(14, GridNav.left(15,  S));
        assertEquals(11, GridNav.above(15, S));
    }

    // --- grille 8x8 : vérifier que squareSize est bien paramétrable ---

    @Test
    public void grille8x8_left_bordGauche_selfLoop() {
        assertEquals(8, GridNav.left(8, 8));   // début de ligne 1
        assertEquals(16, GridNav.left(16, 8)); // début de ligne 2
    }

    @Test
    public void grille8x8_right_milieu() {
        assertEquals(10, GridNav.right(9, 8));
    }
}
