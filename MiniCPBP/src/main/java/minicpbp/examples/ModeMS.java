package minicpbp.examples;
import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;

public class ModeMS implements CPMode {
    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget) {
        if (goalReward > 0) {
            totalReward.removeBelow(goalReward);
        } else if (goalReward == 0 && holeReward == 0) {
            totalReward.assign(0);
        }
    }
}