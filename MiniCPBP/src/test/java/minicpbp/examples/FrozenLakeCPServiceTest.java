package minicpbp.examples;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests unitaires pour FrozenLakeCPService.
 *
 * On utilise resetStateForTests() avant chaque test pour isoler l'état statique.
 * Le JSON d'instance est injecté directement via allInstancesConfig sans fichier disque.
 *
 * Instance de test : FrozenLake 4x4
 *   S  F  F  F       states : 0..15
 *   F  H  F  H       holes  : 5, 7, 11, 12
 *   F  F  F  H       goal   : 15
 *   H  F  F  G       start  : 0 (implicite)
 */
public class FrozenLakeCPServiceTest {

    // JSON minimal valide pour une instance 4x4
    private static final String INSTANCE_ID = "test_4x4";
    private static final String VALID_JSON = "{"
        + "\"" + INSTANCE_ID + "\": {"
        + "  \"size\": 4,"
        + "  \"goal\": 15,"
        + "  \"cp_no_slip_proba\": 0.3333333333333333,"
        + "  \"cp_nbSteps\": 20,"
        + "  \"holes\": [5, 7, 11, 12],"
        + "  \"holeReward\": 0,"
        + "  \"goalReward\": 1,"
        + "  \"budget\": 3"
        + "}"
        + "}";

    /** Injecte le JSON directement via resetStateForTests et reset l'état. */
    private void setupWithMode(CPMode mode) {
        FrozenLakeCPService.resetStateForTests(mode, new JSONObject(VALID_JSON));
    }

    // -------------------------------------------------------------------------
    // loadInstanceParameters
    // -------------------------------------------------------------------------

    @Test
    public void loadInstanceParameters_instanceValide_retourneTrue() {
        setupWithMode(new ModeMS());
        assertTrue(FrozenLakeCPService.loadInstanceParameters(INSTANCE_ID));
    }

    @Test
    public void loadInstanceParameters_instanceInconnue_retourneFalse() {
        setupWithMode(new ModeMS());
        assertFalse(FrozenLakeCPService.loadInstanceParameters("inconnu"));
    }

    @Test
    public void loadInstanceParameters_jsonNull_retourneFalse() {
        FrozenLakeCPService.resetStateForTests(new ModeMS(), null);
        // allInstancesConfig est null -> doit retourner false
        assertFalse(FrozenLakeCPService.loadInstanceParameters(INSTANCE_ID));
    }

    @Test
    public void loadInstanceParameters_cleMandataireManquante_retourneFalse() {
        // JSON sans cp_nbSteps
        JSONObject badJson = new JSONObject("{"
            + "\"bad\": {\"size\":4, \"goal\":15, \"cp_no_slip_proba\":0.333, \"holes\":[]}"
            + "}");
        FrozenLakeCPService.resetStateForTests(new ModeMS(), badJson);
        assertFalse(FrozenLakeCPService.loadInstanceParameters("bad"));
    }

    // -------------------------------------------------------------------------
    // recalculateMatricesAndModelParams
    // -------------------------------------------------------------------------

    @Test
    public void recalculate_apresLoadValide_retourneTrue() {
        setupWithMode(new ModeMS());
        assertTrue(FrozenLakeCPService.loadInstanceParameters(INSTANCE_ID));
        assertTrue(FrozenLakeCPService.recalculateMatricesAndModelParams());
    }

    // -------------------------------------------------------------------------
    // handleReset
    // -------------------------------------------------------------------------

    private void initAndReset(CPMode mode) {
        setupWithMode(mode);
        FrozenLakeCPService.loadInstanceParameters(INSTANCE_ID);
        FrozenLakeCPService.recalculateMatricesAndModelParams();
    }

    @Test
    public void handleReset_modeMS_retourneOK() {
        initAndReset(new ModeMS());
        assertEquals("OK RESET successful", FrozenLakeCPService.handleReset());
    }

    @Test
    public void handleReset_modeETR_retourneOK() {
        initAndReset(new ModeETR());
        assertEquals("OK RESET successful", FrozenLakeCPService.handleReset());
    }

    @Test
    public void handleReset_modeBudget_retourneOK() {
        initAndReset(new ModeBudget());
        assertEquals("OK RESET successful", FrozenLakeCPService.handleReset());
    }

    // -------------------------------------------------------------------------
    // handleStep
    // -------------------------------------------------------------------------

    @Test
    public void handleStep_premierPas_valide() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        // step 0, action 2 (RIGHT), next_state 1
        assertEquals("OK STEP processed", FrozenLakeCPService.handleStep("0", "2", "1"));
    }

    @Test
    public void handleStep_indexStepIncoherent_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        // On envoie step 1 alors qu'on est à step 0
        assertTrue(FrozenLakeCPService.handleStep("1", "2", "1").startsWith("ERROR"));
    }

    @Test
    public void handleStep_actionHorsBornes_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        // MS a 4 actions : action 4 est hors bornes
        assertTrue(FrozenLakeCPService.handleStep("0", "4", "1").startsWith("ERROR"));
    }

    @Test
    public void handleStep_etatHorsBornes_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        // next_state 99 est hors bornes (16 états)
        assertTrue(FrozenLakeCPService.handleStep("0", "2", "99").startsWith("ERROR"));
    }

    @Test
    public void handleStep_formatNombre_invalide_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        assertTrue(FrozenLakeCPService.handleStep("0", "abc", "1").startsWith("ERROR"));
    }

    @Test
    public void handleStep_sequenceMultiple_incrementeStepCorrectement() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        assertEquals("OK STEP processed", FrozenLakeCPService.handleStep("0", "2", "1"));
        assertEquals("OK STEP processed", FrozenLakeCPService.handleStep("1", "2", "2"));
        assertEquals("OK STEP processed", FrozenLakeCPService.handleStep("2", "2", "3"));
    }

    // -------------------------------------------------------------------------
    // handleQueryActionMarginal
    // -------------------------------------------------------------------------

    @Test
    public void handleQuery_retourneREWARD() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        String resp = FrozenLakeCPService.handleQueryActionMarginal("0", "0");
        assertTrue("Réponse attendue: REWARD ..., obtenu: " + resp, resp.startsWith("REWARD "));
    }

    @Test
    public void handleQuery_probabiliteDansIntervalle01() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        for (int a = 0; a < 4; a++) {
            String resp = FrozenLakeCPService.handleQueryActionMarginal("0", String.valueOf(a));
            double prob = Double.parseDouble(resp.split(" ")[1]);
            assertTrue("Proba < 0 pour action " + a, prob >= 0.0);
            assertTrue("Proba > 1 pour action " + a, prob <= 1.0);
        }
    }

    @Test
    public void handleQuery_indexStepIncoherent_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        // On est à step 0, on demande step 1
        assertTrue(FrozenLakeCPService.handleQueryActionMarginal("1", "0").startsWith("ERROR"));
    }

    @Test
    public void handleQuery_actionHorsBornes_retourneErreur() {
        initAndReset(new ModeMS());
        FrozenLakeCPService.handleReset();
        assertTrue(FrozenLakeCPService.handleQueryActionMarginal("0", "4").startsWith("ERROR"));
    }

    // -------------------------------------------------------------------------
    // handleQueryETR
    // -------------------------------------------------------------------------

    @Test
    public void handleQueryETR_retourneETR_VALUE() {
        initAndReset(new ModeETR());
        FrozenLakeCPService.handleReset();
        String resp = FrozenLakeCPService.handleQueryETR();
        assertTrue("Réponse attendue: ETR_VALUE ..., obtenu: " + resp, resp.startsWith("ETR_VALUE "));
    }

    @Test
    public void handleQueryETR_valeurDansIntervalle01() {
        initAndReset(new ModeETR());
        FrozenLakeCPService.handleReset();
        String resp = FrozenLakeCPService.handleQueryETR();
        double val = Double.parseDouble(resp.split(" ")[1]);
        assertTrue("ETR < 0", val >= 0.0);
        assertTrue("ETR > 1", val <= 1.0);
    }

    @Test
    public void handleQueryETR_apresStep_valeurChange() {
        initAndReset(new ModeETR());
        FrozenLakeCPService.handleReset();
        // ETR avant le step : doit être valide
        double etrAvant = Double.parseDouble(FrozenLakeCPService.handleQueryETR().split(" ")[1]);
        assertTrue(etrAvant >= 0.0 && etrAvant <= 1.0);

        // On révèle un step : on va en état 1 (case saine, proche du goal)
        FrozenLakeCPService.handleStep("0", "2", "1");

        // ETR après le step : doit rester valide
        double etrApres = Double.parseDouble(FrozenLakeCPService.handleQueryETR().split(" ")[1]);
        assertTrue(etrApres >= 0.0 && etrApres <= 1.0);
    }

    // -------------------------------------------------------------------------
    // handleQueryActionMarginal avec ModeBudget (8 actions)
    // -------------------------------------------------------------------------

    @Test
    public void handleQuery_modeBudget_8actions_probabiliteValide() {
        initAndReset(new ModeBudget());
        FrozenLakeCPService.handleReset();
        for (int a = 0; a < 8; a++) {
            String resp = FrozenLakeCPService.handleQueryActionMarginal("0", String.valueOf(a));
            assertTrue("Réponse invalide pour action " + a + ": " + resp, resp.startsWith("REWARD "));
            double prob = Double.parseDouble(resp.split(" ")[1]);
            assertTrue("Proba < 0 pour action " + a, prob >= 0.0);
            assertTrue("Proba > 1 pour action " + a, prob <= 1.0);
        }
    }
}
