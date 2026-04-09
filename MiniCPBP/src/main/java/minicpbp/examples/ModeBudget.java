package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import static minicpbp.cp.Factory.*;

public class ModeBudget implements CPMode {
    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget) {
        
        // 1. On garde la contrainte MS classique 
        if (goalReward > 0) {
            totalReward.removeBelow(goalReward);
        }
        
        // 2. Contrainte de budget sur les actions "NoSlip" (valeurs 4, 5, 6, 7)
        int[] noSlipActions = {4, 5, 6, 7};
        
        // On impose qu'au maximum "budget" variables dans 'action' prennent une valeur de V
        cp.post(atmost(action, noSlipActions, budget));
        
        System.out.println("-> Contraintes BUDGET appliquées (Max " + budget + " mouvements sûrs).");
    }
}