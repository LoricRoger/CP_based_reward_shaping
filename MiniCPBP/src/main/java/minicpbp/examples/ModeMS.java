package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;

/**
 * MS mode (Maximize Success).
 *
 * Constrains totalReward to be at least goalReward, forcing the CP model
 * to focus only on trajectories that reach the goal.
 * When budget > 0, also posts an atmost constraint on no-slip actions (4-7).
 *
 * Query: QUERY <step> <action> → marginal probability of action at given step.
 *
 * Note: this mode is secondary and may be removed in future versions.
 */
public class ModeMS extends AbstractCPMode {

    public ModeMS() {
        this(0);
    }

    public ModeMS(int budget) {
        super(budget);
    }

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward) {
        applyBudgetConstraint(cp, action);
        if (goalReward > 0) {
            totalReward.removeBelow(goalReward);
        } else if (goalReward == 0 && holeReward == 0) {
            totalReward.assign(0);
        }
    }
}
