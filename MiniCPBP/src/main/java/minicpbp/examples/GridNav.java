package minicpbp.examples;

/** Static helpers for grid navigation on a square grid. */
class GridNav {
    private GridNav() {}

    static int left(int pos, int squareSize) {
        return (pos % squareSize > 0 ? pos - 1 : pos);
    }

    static int right(int pos, int squareSize) {
        return (pos % squareSize < squareSize - 1 ? pos + 1 : pos);
    }

    static int above(int pos, int squareSize) {
        return (pos >= squareSize ? pos - squareSize : pos);
    }

    static int below(int pos, int squareSize, int nbStates) {
        return (pos < nbStates - squareSize ? pos + squareSize : pos);
    }
}
