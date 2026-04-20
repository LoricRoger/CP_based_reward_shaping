package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;

/**
 * ETR mode (Expected Total Reward / Success Probability).
 *
 * Explores all trajectories without restricting totalReward.
 * When budget > 0, posts an atmost constraint to limit no-slip actions (4-7).
 *
 * Query: QUERY_ETR → marginal of goal state at the last step.
 */
public class ModeETR extends AbstractCPMode {

    public ModeETR() {
        this(0);
    }

    public ModeETR(int budget) {
        super(budget);
    }

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward) {
        // ETR explores all possibilities — no constraint on totalReward.
        applyBudgetConstraint(cp, action);
    }
}
