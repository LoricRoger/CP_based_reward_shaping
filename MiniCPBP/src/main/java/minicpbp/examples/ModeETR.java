package minicpbp.examples;
import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;

public class ModeETR implements CPMode {
    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward, int budget) {
        // En ETR, on explore toutes les possibilités
    }
}