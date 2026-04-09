// FrozenLakeCPService.java
package minicpbp.examples;

// MiniCPBP imports
import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;
import static minicpbp.cp.Factory.*;

// Java IO and Net imports
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;

// Data structures
import java.util.*;

// JSON Library Import
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;


public class FrozenLakeCPService {

    private static final int PORT = 12345;
    private static final String INSTANCES_JSON_FILE = "instances.json";

    private static int squareSize = -1;
    private static int nbStates = -1;
    private static int nbSteps = -1;
    private static double noSlipProba = Double.NaN;
    private static double sideSlipProba = Double.NaN;
    private static int holeReward = 0;
    private static int goalReward = 1;
    private static int[] holes = null;
    private static int nbActions = -1;
    private static int noSlipBudget = -1;

    private static double[][][] P_matrix;
    private static int[][][] R_matrix;
    @SuppressWarnings("unused") // TODO: à utiliser pour les contraintes de terminaison
    private static List<Integer> acceptingStates; // States where episode can end (goal + holes)
    private static Set<Integer> holeSet = new HashSet<>();
    private static int goalStateIdx = -1;

    private static Solver cp;
    private static IntVar[] action;
    private static IntVar[] state;
    private static IntVar totalReward; // Represents SUM of rewards over nbSteps horizon
    private static int currentEpisodeStep = 0; // Tracks current step within the CP model horizon

    private static JSONObject allInstancesConfig = null;
    private static String currentInstanceId = null;

    private static final int BP_ITERATIONS = 1; // Number of belief propagation iterations
    private static CPMode currentMode;

    public static void main(String[] args) {
        String modeArg = (args.length > 0) ? args[0].toUpperCase() : "MS";

        switch (modeArg) {
            case "MS":
                currentMode = new ModeMS();
                break;
            case "ETR":
                currentMode = new ModeETR();
                break;
            case "BUDGET":
                currentMode = new ModeBudget();
                break;
            default:
                System.err.println("FATAL: Mode inconnu '" + modeArg + "'. Utilisez MS, ETR ou BUDGET.");
                System.exit(1);
        }
        System.out.println("Serveur FrozenLake démarré en mode : " + modeArg);

        if (!loadAllInstancesConfig()) {
            System.err.println("FATAL: Could not load instances configuration from '" + INSTANCES_JSON_FILE + "'. Exiting.");
            System.exit(1);
        }
        System.out.println("Successfully loaded instance configurations from " + INSTANCES_JSON_FILE);

        runServer();
    }

    static void runServer() {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("FrozenLake CP Server listening on port " + PORT);

            while (true) {
                try (Socket clientSocket = serverSocket.accept();
                     PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {

                    System.out.println("\nClient connected: " + clientSocket.getInetAddress());
                    out.println("OK Welcome");

                    currentInstanceId = null;
                    cp = null;

                    String inputLine;
                    while ((inputLine = in.readLine()) != null) {
                        String[] tokens = inputLine.trim().split("\\s+");
                        String command = tokens.length > 0 ? tokens[0].toUpperCase() : "";
                        String response = "ERROR Unknown command '" + command + "'";

                        try {
                            switch (command) {
                                case "INIT":
                                    if (tokens.length == 2) {
                                        String reqId = tokens[1];
                                        if (loadInstanceParameters(reqId) && recalculateMatricesAndModelParams()) {
                                            currentInstanceId = reqId;
                                            response = "OK INIT successful for " + currentInstanceId;
                                            System.out.println("Initialized for instance: " + currentInstanceId);
                                        } else {
                                            response = "ERROR Failed loading/recalculating for instance " + reqId;
                                            System.err.println(response);
                                            currentInstanceId = null;
                                        }
                                    } else {
                                        response = "ERROR Invalid INIT format. Expected: INIT <instance_id>";
                                    }
                                    break;

                                case "RESET":
                                    if (currentInstanceId == null) {
                                        response = "ERROR Must INIT first";
                                    } else {
                                        response = handleReset();
                                    }
                                    break;

                                case "STEP":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 4) {
                                        response = handleStep(tokens[1], tokens[2], tokens[3]);
                                    } else {
                                        response = "ERROR Invalid STEP format. Expected: STEP <step_idx> <action_idx> <next_state_idx>";
                                    }
                                    break;

                                case "QUERY":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 3) {
                                        response = handleQueryActionMarginal(tokens[1], tokens[2]);
                                    } else {
                                        response = "ERROR Invalid QUERY format. Expected: QUERY <step_idx> <action_idx>";
                                    }
                                    break;

                                case "QUERY_ETR":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 1) {
                                        response = handleQueryETR();
                                    } else {
                                        response = "ERROR Invalid QUERY_ETR format. Expected: QUERY_ETR";
                                    }
                                    break;

                                case "QUIT":
                                    response = "OK Goodbye";
                                    System.out.println("Client requested QUIT.");
                                    break;

                                default:
                                    System.out.println("Received unknown command: " + inputLine);
                                    break;
                            }
                        } catch (Exception e) {
                            System.err.println("Critical Error processing client command '" + inputLine + "': " + e.getMessage());
                            e.printStackTrace();
                            response = "ERROR Processing failed: " + e.getClass().getSimpleName();
                        }

                        out.println(response);

                        if ("QUIT".equalsIgnoreCase(command)) {
                            break;
                        }
                    }
                } catch (IOException e) {
                    System.err.println("WARN: Client connection error: " + e.getMessage());
                } finally {
                    System.out.println("Client disconnected.");
                    currentInstanceId = null;
                    cp = null;
                }
            }
        } catch (IOException e) {
            System.err.println("FATAL: Server socket error on port " + PORT + ": " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    static boolean loadAllInstancesConfig() {
        try {
            String jsonContent = new String(Files.readAllBytes(Paths.get(INSTANCES_JSON_FILE)));
            allInstancesConfig = new JSONObject(jsonContent);
            return true;
        } catch (IOException e) {
            System.err.println("ERROR reading instances file '" + INSTANCES_JSON_FILE + "': " + e.getMessage());
            System.err.println("  Current working directory: " + Paths.get(".").toAbsolutePath().normalize().toString());
            return false;
        } catch (JSONException e) {
            System.err.println("ERROR parsing JSON from '" + INSTANCES_JSON_FILE + "': " + e.getMessage());
            return false;
        } catch (Exception e) {
            System.err.println("ERROR loading JSON config: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    static boolean loadInstanceParameters(String instanceId) {
        if (allInstancesConfig == null) {
            System.err.println("ERROR: Instance configurations JSON not loaded.");
            return false;
        }
        if (!allInstancesConfig.has(instanceId)) {
            System.err.println("ERROR: Instance ID '" + instanceId + "' not found in " + INSTANCES_JSON_FILE);
            return false;
        }
        try {
            JSONObject d = allInstancesConfig.getJSONObject(instanceId);
            squareSize = d.getInt("size");
            nbStates = squareSize * squareSize;
            goalStateIdx = d.getInt("goal");
            noSlipProba = d.getDouble("cp_no_slip_proba");
            sideSlipProba = (1.0 - noSlipProba) / 2.0;
            holeReward = d.optInt("holeReward", 0);
            goalReward = d.optInt("goalReward", 1);
            noSlipBudget = d.optInt("budget", 0);
            nbSteps = d.getInt("cp_nbSteps");

            JSONArray hArray = d.getJSONArray("holes");
            List<Integer> vHoles = new ArrayList<>();
            for (int i = 0; i < hArray.length(); i++) {
                int h = hArray.getInt(i);
                if (h >= 0 && h < nbStates && h != 0 && h != goalStateIdx) {
                    vHoles.add(h);
                } else {
                    System.err.println("WARN: Instance " + instanceId + " has invalid hole index " + h + ". Ignoring.");
                }
            }
            holes = vHoles.stream().mapToInt(Integer::intValue).toArray();

            System.out.println("Loaded params for '" + instanceId + "': Size=" + squareSize +
                               ", Goal=" + goalStateIdx + ", NoSlip=" + String.format("%.3f", noSlipProba) +
                               ", Holes=" + Arrays.toString(holes) + ", CP_Steps=" + nbSteps +
                               ", Budget=" + noSlipBudget);
            return true;
        } catch (JSONException e) {
            System.err.println("ERROR: JSON parse error loading params for '" + instanceId + "': " + e.getMessage());
            if (e.getMessage() != null && e.getMessage().contains("cp_nbSteps")) {
                System.err.println(">>> Ensure 'cp_nbSteps' key exists and is an integer in '" + instanceId + "' config.");
            }
            if (e.getMessage() != null && e.getMessage().contains("goal")) {
                System.err.println(">>> Ensure 'goal' key exists and is an integer in '" + instanceId + "' config.");
            }
            return false;
        } catch (Exception e) {
            System.err.println("ERROR loading params for '" + instanceId + "': " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    static boolean recalculateMatricesAndModelParams() {
        try {
            nbActions = currentMode.getNbActions();
            holeSet.clear();
            for (int h : holes) {
                holeSet.add(h);
            }
            acceptingStates = calculateAcceptingStates();
            P_matrix = calculatePMatrix();
            R_matrix = calculateRMatrix();
            System.out.println("Recalculated P/R matrices and model parameters.");
            return true;
        } catch (Exception e) {
            System.err.println("FATAL: Exception during matrix recalculation.");
            e.printStackTrace();
            return false;
        }
    }

    static String handleReset() {
        System.out.println("Handling RESET command...");
        String response;
        try {
            cp = makeSolver();

            action = makeIntVarArray(cp, nbSteps, nbActions);
            state = makeIntVarArray(cp, nbSteps, nbStates);
            int minR = Math.min(0, nbSteps * holeReward);
            int maxR = Math.max(0, nbSteps * goalReward);
            totalReward = makeIntVar(cp, minR, maxR);

            Constraint c = markov(action, state, P_matrix, R_matrix, 0, totalReward);
            cp.post(c);

            currentMode.applyConstraints(cp, action, totalReward, goalReward, holeReward, noSlipBudget);

            currentEpisodeStep = 0;
            cp.fixPoint();
            System.out.println("CP Model Reset successfully.");
            response = "OK RESET successful";

        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            response = "ERROR RESET failed: " + e.getMessage();
            cp = null;
        }
        return response;
    }

    static String handleStep(String iStr, String aStr, String sNextStr) {
        if (cp == null) return "ERROR Must RESET first";
        try {
            int i = Integer.parseInt(iStr);
            int a = Integer.parseInt(aStr);
            int sN = Integer.parseInt(sNextStr);

            if (i != currentEpisodeStep) {
                System.err.println("WARN: STEP index mismatch. Expected " + currentEpisodeStep + ", got " + i);
                return "ERROR STEP index mismatch. Expected " + currentEpisodeStep;
            }
            if (i < 0 || i >= nbSteps) return "ERROR Step index " + i + " out of bounds [0.." + (nbSteps - 1) + "]";
            if (a < 0 || a >= nbActions) return "ERROR Action index " + a + " out of bounds [0.." + (nbActions - 1) + "]";
            if (sN < 0 || sN >= nbStates) return "ERROR Next state index " + sN + " out of bounds [0.." + (nbStates - 1) + "]";

            action[i].assign(a);
            state[i].assign(sN);
            cp.fixPoint();
            currentEpisodeStep++;
            return "OK STEP processed";

        } catch (InconsistencyException e) {
            System.err.println("ERROR: Inconsistency detected on STEP: " + e.getMessage());
            return "ERROR Inconsistency STEP " + iStr;
        } catch (NumberFormatException e) {
            System.err.println("ERROR Invalid number in STEP command: " + e.getMessage());
            return "ERROR Invalid number in STEP command: " + e.getMessage();
        } catch (Exception e) {
            System.err.println("ERROR Unexpected failure processing STEP: " + e.getMessage());
            e.printStackTrace();
            return "ERROR Unexpected failure processing STEP: " + e.getMessage();
        }
    }

    static String handleQueryActionMarginal(String iStr, String aQueryStr) {
        if (cp == null) return "ERROR Must RESET first";
        try {
            int i = Integer.parseInt(iStr);
            int aQ = Integer.parseInt(aQueryStr);

            if (i != currentEpisodeStep) return "ERROR QUERY index mismatch. Expected " + currentEpisodeStep;
            if (i < 0 || i >= nbSteps) return "ERROR Step index " + i + " out of bounds";
            if (aQ < 0 || aQ >= nbActions) return "ERROR Action index " + aQ + " out of bounds";

            cp.vanillaBP(BP_ITERATIONS);
            cp.fixPoint();

            double prob = action[i].marginal(aQ);

            if (Double.isNaN(prob) || prob < -1e-9 || prob > 1.0 + 1e-9) {
                System.err.println("WARN: Invalid marginal probability " + prob + " for Action " + aQ + " at Step " + i + ". Clamping to 0.");
                prob = 0.0;
            } else {
                prob = Math.max(0.0, Math.min(1.0, prob));
            }
            return "REWARD " + prob;

        } catch (InconsistencyException e) {
            System.err.println("WARN: Inconsistency detected during propagation for QUERY Step. Returning 0.0.");
            return "REWARD 0.0";
        } catch (NumberFormatException e) {
            System.err.println("ERROR Invalid number in QUERY command: " + e.getMessage());
            return "REWARD 0.0";
        } catch (Exception e) {
            System.err.println("ERROR getting marginal: " + e.getMessage());
            e.printStackTrace();
            return "REWARD 0.0";
        }
    }

    static String handleQueryETR() {
        if (cp == null) return "ERROR Must RESET first";
        try {
            cp.vanillaBP(BP_ITERATIONS);
            cp.fixPoint();

            double etrValue = totalReward.marginal(goalReward);

            if (Double.isNaN(etrValue) || etrValue < -1e-9 || etrValue > 1.0 + 1e-9) {
                System.err.println("WARN: Invalid ETR value " + etrValue + " obtained. Clamping to 0.");
                etrValue = 0.0;
            } else {
                etrValue = Math.max(0.0, Math.min(1.0, etrValue));
            }
            System.err.println("ETR calculé : " + etrValue);
            return "ETR_VALUE " + etrValue;

        } catch (InconsistencyException e) {
            System.err.println("WARN: Inconsistency detected during propagation for QUERY_ETR. Returning 0.0.");
            return "ETR_VALUE 0.0";
        } catch (Exception e) {
            System.err.println("ERROR getting ETR value: " + e.getMessage());
            e.printStackTrace();
            return "ETR_VALUE 0.0";
        }
    }

    private static List<Integer> calculateAcceptingStates() {
        List<Integer> states = new ArrayList<>();
        if (goalStateIdx >= 0) states.add(goalStateIdx);
        holeSet.forEach(h -> states.add(h));
        return states;
    }

    private static double[][][] calculatePMatrix() {
        double[][][] P = new double[nbStates][nbActions][nbStates];
        currentMode.fillTransitions(P, nbStates, squareSize, holeSet, goalStateIdx, noSlipProba, sideSlipProba);
        return P;
    }

    private static int[][][] calculateRMatrix() {
        int[][][] R = new int[nbStates][nbActions][nbStates];
        for (int i = 0; i < nbStates; i++) {
            if (holeSet.contains(i) || i == goalStateIdx) continue;
            for (int j = 0; j < nbActions; j++) {
                for (int k = 0; k < nbStates; k++) {
                    if (k == goalStateIdx) {
                        R[i][j][k] = goalReward;
                    } else if (holeSet.contains(k)) {
                        R[i][j][k] = holeReward;
                    }
                }
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // TEST SUPPORT ONLY — ne pas appeler en production.
    // Remet à zéro tout l'état statique entre deux tests unitaires.
    // Nécessaire car les champs static fuiteraient d'un test à l'autre.
    // -------------------------------------------------------------------------
    static void resetStateForTests(CPMode mode, org.json.JSONObject instancesConfig) {
        currentMode        = mode;
        allInstancesConfig = instancesConfig;
        currentInstanceId = null;
        squareSize       = -1;
        nbStates         = -1;
        nbSteps          = -1;
        noSlipProba      = Double.NaN;
        sideSlipProba    = Double.NaN;
        holeReward       = 0;
        goalReward       = 1;
        holes            = null;
        nbActions        = -1;
        noSlipBudget     = -1;
        P_matrix         = null;
        R_matrix         = null;
        acceptingStates  = null;
        holeSet          = new HashSet<>();
        goalStateIdx     = -1;
        cp               = null;
        action           = null;
        state            = null;
        totalReward      = null;
        currentEpisodeStep = 0;
    }

} // End of class FrozenLakeCPService
