// FrozenLakeCPService.java
package minicpbp.examples;

// MiniCPBP imports
import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;
import static minicpbp.cp.Factory.*; // Import static factory methods

// Java IO and Net imports
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths; // Used for reading file by path

// Data structures
import java.util.*;
import java.util.stream.Collectors;

// JSON Library Import
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;


public class FrozenLakeCPService {

    private static final int PORT = 12345;
    // Assume instances.json is in the same directory or accessible from where java is run
    private static final String INSTANCES_JSON_FILE = "instances.json";

    private static int squareSize = 4;
    private static int nbStates = squareSize * squareSize;
    private static int nbSteps = 110; // Default CP steps (will be overwritten by JSON)
    private static double noSlipProba = 0.3333333333333333;
    private static double sideSlipProba = (1.0 - noSlipProba) / 2.0;
    private static int holeReward = 0;
    private static int goalReward = 1;
    private static int[] holes = {5, 7, 11, 12}; // Default 4x4 holes

    private static double[][][] P_matrix;
    private static int[][][] R_matrix;
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
            case "ETR" :
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

        // Load instance configurations once at the start
        if (!loadAllInstancesConfig()) {
            System.err.println("FATAL: Could not load instances configuration from '" + INSTANCES_JSON_FILE + "'. Exiting.");
            System.exit(1);
        }
        System.out.println("Successfully loaded instance configurations from " + INSTANCES_JSON_FILE);

        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("FrozenLake CP Server listening on port " + PORT);

            while (true) { // Keep accepting client connections
                try (Socket clientSocket = serverSocket.accept();
                     PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {

                    System.out.println("\nClient connected: " + clientSocket.getInetAddress());
                    out.println("OK Welcome"); // Send welcome message

                    currentInstanceId = null; // Reset instance for new client
                    cp = null; // Reset solver for new client

                    String inputLine;
                    // Process commands from the client until QUIT or disconnect
                    while ((inputLine = in.readLine()) != null) {
                        String[] tokens = inputLine.trim().split("\\s+");
                        String command = tokens.length > 0 ? tokens[0].toUpperCase() : "";
                        String response = "ERROR Unknown command '" + command + "'"; // Default error

                        try {
                            switch (command) {
                                case "INIT":
                                    if (tokens.length == 2) {
                                        String reqId = tokens[1];
                                        // Attempt to load parameters and build model
                                        if (loadInstanceParameters(reqId) && recalculateMatricesAndModelParams()) {
                                            currentInstanceId = reqId;
                                            response = "OK INIT successful for " + currentInstanceId;
                                            System.out.println("Initialized for instance: " + currentInstanceId);
                                        } else {
                                            response = "ERROR Failed loading/recalculating for instance " + reqId;
                                            System.err.println(response);
                                            currentInstanceId = null; // Ensure reset on failure
                                        }
                                    } else {
                                        response = "ERROR Invalid INIT format. Expected: INIT <instance_id>";
                                    }
                                    break;

                                case "RESET":
                                    if (currentInstanceId == null) {
                                        response = "ERROR Must INIT first";
                                    } else {
                                        response = handleReset(); // Rebuilds the CP model
                                    }
                                    break;

                                case "STEP":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 4) {
                                        response = handleStep(tokens[1], tokens[2], tokens[3]); // Process step
                                    } else {
                                        response = "ERROR Invalid STEP format. Expected: STEP <step_idx> <action_idx> <next_state_idx>";
                                    }
                                    break;

                                case "QUERY": // For cp-ms (action marginals)
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 3) {
                                        response = handleQueryActionMarginal(tokens[1], tokens[2]);
                                    } else {
                                        response = "ERROR Invalid QUERY format. Expected: QUERY <step_idx> <action_idx>";
                                    }
                                    break;

                                case "QUERY_ETR": // New command for cp-etr (Expected Total Reward)
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 1) { // No extra args needed
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
                                     // Keep default error response
                                     break;
                            }
                        } catch (Exception e) {
                            // Catch unexpected errors during command processing
                            System.err.println("Critical Error processing client command '" + inputLine + "': " + e.getMessage());
                            e.printStackTrace();
                            response = "ERROR Processing failed: " + e.getClass().getSimpleName();
                        }

                        out.println(response); // Send response back to client

                        if ("QUIT".equalsIgnoreCase(command)) { // Exit loop if QUIT command was received
                            break;
                        }
                    }
                } catch (IOException e) {
                    System.err.println("WARN: Client connection error: " + e.getMessage());
                    // Client likely disconnected abruptly
                } finally {
                    System.out.println("Client disconnected.");
                    currentInstanceId = null; // Clean up state for next potential client
                    cp = null;
                }
            }
        } catch (IOException e) {
            System.err.println("FATAL: Server socket error on port " + PORT + ": " + e.getMessage());
            e.printStackTrace();
            System.exit(1); // Exit if server cannot bind to port
        }
    }

    private static boolean loadAllInstancesConfig() {
        try {
            // Try to read relative to current working directory
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

    private static boolean loadInstanceParameters(String instanceId) {
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
            goalStateIdx = d.getInt("goal"); // Assuming goal is always present
            noSlipProba = d.getDouble("cp_no_slip_proba");
            sideSlipProba = (1.0 - noSlipProba) / 2.0;
            holeReward = d.optInt("holeReward", 0); // Use defaults if missing
            goalReward = d.optInt("goalReward", 1);

            nbSteps = d.getInt("cp_nbSteps"); // Use the CP-specific step count

            JSONArray hArray = d.getJSONArray("holes");
            List<Integer> vHoles = new ArrayList<>();
            for (int i = 0; i < hArray.length(); i++) {
                int h = hArray.getInt(i);
                // Validate hole index
                if (h >= 0 && h < nbStates && h != 0 && h != goalStateIdx) { // Cannot be start or goal
                    vHoles.add(h);
                } else {
                    System.err.println("WARN: Instance " + instanceId + " has invalid hole index " + h + ". Ignoring.");
                }
            }
            holes = vHoles.stream().mapToInt(Integer::intValue).toArray();

            System.out.println("Loaded params for '" + instanceId + "': Size=" + squareSize +
                               ", Goal=" + goalStateIdx + ", NoSlip=" + String.format("%.3f", noSlipProba) +
                               ", Holes=" + Arrays.toString(holes) + ", CP_Steps=" + nbSteps);
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

    private static boolean recalculateMatricesAndModelParams() {
        try {
            holeSet.clear();
            for (int h : holes) { // Update holeSet
                holeSet.add(h);
            }
            acceptingStates = calculateAcceptingStates(); // Goal + Holes
            P_matrix = calculatePMatrix(); // Calculate transitions
            R_matrix = calculateRMatrix(); // Calculate immediate rewards
            System.out.println("Recalculated P/R matrices and model parameters.");
            return true;
        } catch (Exception e) {
            System.err.println("FATAL: Exception during matrix recalculation.");
            e.printStackTrace();
            return false;
        }
    }

    private static String handleReset() {
        System.out.println("Handling RESET command...");
        String response;
        try {
            cp = makeSolver(); // Create a new solver instance

            // Define variables for the entire trajectory
            action = makeIntVarArray(cp, nbSteps, nbActions); // Actions at each step
            state = makeIntVarArray(cp, nbSteps, nbStates);   // State *reached* at end of each step
            int minR = Math.min(0, nbSteps * holeReward); // Min possible total reward
            int maxR = Math.max(0, nbSteps * goalReward); // Max possible total reward
            totalReward = makeIntVar(cp, minR, maxR);     // Variable for the sum of rewards

            // Post the core Markov constraint linking actions, states, transitions, and rewards
            Constraint c = markov(action, state, P_matrix, R_matrix, 0, totalReward);
            cp.post(c);

            currentMode.applyConstraints(cp, action, totalReward, goalReward, holeReward, noSplipBudget);


            currentEpisodeStep = 0; // Reset step counter for the CP model horizon
            cp.fixPoint(); // Initial propagation
            System.out.println("CP Model Reset successfully.");
            response = "OK RESET successful";

        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            response = "ERROR RESET failed: " + e.getMessage();
            cp = null; // Ensure solver is nullified on failure
        }
        return response;
    }

    private static String handleStep(String iStr, String aStr, String sNextStr) {
        String response;
        if (cp == null) {
            return "ERROR Must RESET first";
        }
        try {
            int i = Integer.parseInt(iStr);         // Step index
            int a = Integer.parseInt(aStr);         // Action taken at step i
            int sN = Integer.parseInt(sNextStr);    // State reached *after* action a at step i

            // Validate indices
            if (i != currentEpisodeStep) {
                System.err.println("WARN: STEP index mismatch. Expected " + currentEpisodeStep + ", got " + i);
                // Mismatched step indices can indicate issues; returning error for strict state management.
                return "ERROR STEP index mismatch. Expected " + currentEpisodeStep;
            }
            if (i < 0 || i >= nbSteps) {
                return "ERROR Step index " + i + " out of bounds [0.." + (nbSteps - 1) + "]";
            }
            if (a < 0 || a >= nbActions) {
                return "ERROR Action index " + a + " out of bounds [0.." + (nbActions - 1) + "]";
            }
            if (sN < 0 || sN >= nbStates) {
                return "ERROR Next state index " + sN + " out of bounds [0.." + (nbStates - 1) + "]";
            }

            try {
                // Assign the known action and resulting state for step i
                action[i].assign(a);
                state[i].assign(sN); // state[i] is the state *at the end* of step i

                cp.fixPoint(); // Propagate the consequences of this assignment

                currentEpisodeStep++; // Increment the internal step counter
                response = "OK STEP processed";

            } catch (InconsistencyException e) {
                System.err.println("ERROR: Inconsistency detected on STEP " + i + " (A=" + a + ", S_next=" + sN + "). " + e.getMessage());
                // This means the provided step is impossible according to the model's current beliefs.
                response = "ERROR Inconsistency STEP " + i;
            }
        } catch (NumberFormatException e) {
            response = "ERROR Invalid number in STEP command: " + e.getMessage();
            System.err.println(response);
        } catch (Exception e) {
            response = "ERROR Unexpected failure processing STEP: " + e.getMessage();
            System.err.println(response);
            e.printStackTrace();
        }
        return response;
    }

    private static String handleQueryActionMarginal(String iStr, String aQueryStr) {
        String response;
        if (cp == null) {
            return "ERROR Must RESET first";
        }
        double prob = 0.0; // Default probability on error
        try {
            int i = Integer.parseInt(iStr);       // Step index to query
            int aQ = Integer.parseInt(aQueryStr); // Action index to query

            // Validate indices
            if (i != currentEpisodeStep) {
                return "ERROR QUERY index mismatch. Expected " + currentEpisodeStep;
            }
            if (i < 0 || i >= nbSteps) {
                return "ERROR Step index " + i + " out of bounds";
            }
            if (aQ < 0 || aQ >= nbActions) {
                return "ERROR Action index " + aQ + " out of bounds";
            }

            try {
                cp.vanillaBP(BP_ITERATIONS); // Run belief propagation
                cp.fixPoint();               // Propagate constraints

                prob = action[i].marginal(aQ); // Get marginal probability for the queried action

                // Validate marginal probability
                if (Double.isNaN(prob) || prob < -1e-9 || prob > 1.0 + 1e-9) { // Check bounds with tolerance
                    System.err.println("WARN: Invalid marginal probability " + prob + " for Action " + aQ + " at Step " + i + ". Clamping to 0.");
                    prob = 0.0;
                } else {
                    prob = Math.max(0.0, Math.min(1.0, prob)); // Clamp to [0, 1] robustly
                }
                response = "REWARD " + prob; // Return the calculated marginal

            } catch (InconsistencyException e) {
                System.err.println("WARN: Inconsistency detected during propagation for QUERY Step " + i + ". Returning 0.0.");
                prob = 0.0;
                response = "REWARD " + prob;
            } catch (Exception e) {
                 System.err.println("ERROR getting marginal for Step " + i + ", Action " + aQ + ": " + e.getMessage());
                 e.printStackTrace();
                 prob = 0.0;
                 response = "REWARD " + prob; // Return 0 on error
            }
        } catch (NumberFormatException e) {
            response = "ERROR Invalid number in QUERY command: " + e.getMessage();
            System.err.println(response);
            response = "REWARD 0.0"; // Default on format error
        } catch (Exception e) {
             response = "ERROR Unexpected failure processing QUERY: " + e.getMessage();
             System.err.println(response);
             e.printStackTrace();
             response = "REWARD 0.0"; // Default on other errors
        }
        return response;
    }

    private static String handleQueryETR() {
        String response;
        if (cp == null) {
            return "ERROR Must RESET first";
        }
        double etrValue = 0.0; // Default ETR (probability of success) on error
        try {
            try {
                cp.vanillaBP(BP_ITERATIONS); // Run belief propagation
                cp.fixPoint();               // Propagate constraints

                // Get the marginal probability of the total reward being 1 (or >= 1)
                // Since we constrained totalReward >= 1, totalReward.marginal(1) should give P(success)
                // If totalReward could be > 1 (e.g. different reward scheme), we might need marginal(v>=1)
                etrValue = totalReward.marginal(goalReward); // P(totalReward == goalReward)

                // Validate the ETR value (should be a probability)
                if (Double.isNaN(etrValue) || etrValue < -1e-9 || etrValue > 1.0 + 1e-9) {
                    System.err.println("WARN: Invalid ETR value " + etrValue + " obtained. Clamping to 0.");
                    etrValue = 0.0;
                } else {
                     etrValue = Math.max(0.0, Math.min(1.0, etrValue)); // Clamp to [0, 1]
                }
                response = "ETR_VALUE " + etrValue; // Return the calculated ETR value

            } catch (InconsistencyException e) {
                 System.err.println("WARN: Inconsistency detected during propagation for QUERY_ETR. Returning 0.0.");
                 etrValue = 0.0;
                 response = "ETR_VALUE " + etrValue;
            } catch (Exception e) {
                 System.err.println("ERROR getting ETR value: " + e.getMessage());
                 e.printStackTrace();
                 etrValue = 0.0;
                 response = "ETR_VALUE " + etrValue; // Return 0 on error
            }
        } catch (Exception e) {
             response = "ERROR Unexpected failure processing QUERY_ETR: " + e.getMessage();
             System.err.println(response);
             e.printStackTrace();
             response = "ETR_VALUE 0.0"; // Default on other errors
        }
        System.err.println("ETR calculé : " + etrValue);
        return response;
    }

    private static List<Integer> calculateAcceptingStates() {
        // Accepting states are where the episode can naturally end: goal or holes
        List<Integer> states = new ArrayList<>();
        if (goalStateIdx >= 0) {
            states.add(goalStateIdx);
        }
        holeSet.forEach(h -> states.add(h));
        return states;
    }

    // Calculates transition probability matrix P[state][action][next_state]
    private static double[][][] calculatePMatrix() {
        double[][][] P = new double[nbStates][nbActions][nbStates];

        if (nbActions == 4){
        for (int i = 0; i < nbStates; i++) { // Current state i
             // Check if terminal state (goal or hole)
             if (holeSet.contains(i) || i == goalStateIdx) {
                 // Terminal state: force self-loop for all actions
                 for (int j = 0; j < nbActions; j++) {
                     // P[i][j][k] is already 0.0 by default initialization
                     P[i][j][i] = 1.0; // Probability 1 of staying in state i
                 }
             } else {
                 // Non-terminal state: calculate transitions based on slip probabilities
                 for (int j = 0; j < nbActions; j++) { // Action j taken
                     // Determine intended and perpendicular next states
                     int s_intended;
                     int s_perp1;
                     int s_perp2;
                     switch (j) {
                         case 0: s_intended = left(i);  s_perp1 = above(i); s_perp2 = below(i); break; // LEFT
                         case 1: s_intended = below(i); s_perp1 = left(i);  s_perp2 = right(i); break; // DOWN
                         case 2: s_intended = right(i); s_perp1 = above(i); s_perp2 = below(i); break; // RIGHT
                         case 3: s_intended = above(i); s_perp1 = left(i);  s_perp2 = right(i); break; // UP
                         default: // Should not happen
                             System.err.println("FATAL: Illegal action " + j + " in PMatrix calculation.");
                             s_intended = i; s_perp1 = i; s_perp2 = i; // Self-loop on error
                     }

                     // Assign probabilities (handle case where perpendicular states are the same)
                     if (s_perp1 == s_perp2) { // e.g., in a corner, moving towards wall
                          P[i][j][s_intended] += noSlipProba;
                          P[i][j][s_perp1] += 2 * sideSlipProba; // Both slips lead here
                     } else {
                          P[i][j][s_intended] += noSlipProba;
                          P[i][j][s_perp1] += sideSlipProba;
                          P[i][j][s_perp2] += sideSlipProba;
                     }

                     // Normalize probabilities for state i, action j (should sum to 1)
                     double sum_k = 0;
                     for (int k = 0; k < nbStates; k++) {
                         sum_k += P[i][j][k];
                     }
                     if (sum_k > 1e-9) { // Avoid division by zero if sum is effectively zero
                         if (Math.abs(sum_k - 1.0) > 1e-9) { // Check if normalization needed
                             for (int k = 0; k < nbStates; k++) {
                                 P[i][j][k] /= sum_k;
                             }
                         } // else: sum is already close enough to 1.0
                     } else {
                         // This should not happen if actions always lead *somewhere*
                         System.err.println("WARN: Zero probability sum for P[" + i + "][" + j + "]. Forcing self-loop.");
                         P[i][j][i] = 1.0;
                     }
                 }
             }
        }
    }
    else if (nbActions == 8) {
        for (int i = 0; i < nbStates; i++) {
                if (holeSet.contains(i) || i == goalStateIdx) {
                    for (int j = 0; j < nbActions; j++) P[i][j][i] = 1.0;
                } else {
                    for (int j = 0; j < nbActions; j++) {
                        int dir = j % 4; // 0,1,2,3 correspond toujours aux directions
                        boolean isNoSlip = (j >= 4); // Les actions 4,5,6,7 sont magiques
                        
                        int s_intended, s_perp1, s_perp2;
                        switch (dir) {
                            case 0: s_intended = left(i);  s_perp1 = above(i); s_perp2 = below(i); break;
                            case 1: s_intended = below(i); s_perp1 = left(i);  s_perp2 = right(i); break;
                            case 2: s_intended = right(i); s_perp1 = above(i); s_perp2 = below(i); break;
                            case 3: s_intended = above(i); s_perp1 = left(i);  s_perp2 = right(i); break;
                            default: s_intended = i; s_perp1 = i; s_perp2 = i;
                        }

                        if (isNoSlip) {
                            P[i][j][s_intended] += 1.0; // 100% de réussite
                        } else {
                            if (s_perp1 == s_perp2) {
                                P[i][j][s_intended] += noSlipProba;
                                P[i][j][s_perp1] += 2 * sideSlipProba;
                            } else {
                                P[i][j][s_intended] += noSlipProba;
                                P[i][j][s_perp1] += sideSlipProba;
                                P[i][j][s_perp2] += sideSlipProba;
                            }
                        }
                        
                        double sum_k = 0;
                        for (int k = 0; k < nbStates; k++) sum_k += P[i][j][k];
                        if (sum_k > 1e-9 && Math.abs(sum_k - 1.0) > 1e-9) {
                            for (int k = 0; k < nbStates; k++) P[i][j][k] /= sum_k;
                        } else if (sum_k <= 1e-9) P[i][j][i] = 1.0;
                    }
                }
            }
        }
        return P;
    }

    // Calculates immediate reward matrix R[state][action][next_state]
    private static int[][][] calculateRMatrix() {
        int[][][] R = new int[nbStates][nbActions][nbStates]; // Initializes to 0
        for (int i = 0; i < nbStates; i++) { // From state i
            if (holeSet.contains(i) || i == goalStateIdx) { // No rewards for transitions *from* a terminal state
                continue;
            }

            for (int j = 0; j < nbActions; j++) { // Taking action j
                 // Reward depends only on the destination state k
                 for (int k = 0; k < nbStates; k++) { // Potential destination state k
                     if (k == goalStateIdx) {
                         R[i][j][k] = goalReward; // Reward for landing in goal state k
                     } else if (holeSet.contains(k)) {
                         R[i][j][k] = holeReward; // Reward for landing in hole state k
                     } // Otherwise R[i][j][k] remains 0 (no reward for landing on normal ice)
                 }
            }
        }
        return R;
    }

    // Grid navigation helpers (assuming square grid)
    private static int left(int pos) { return (pos % squareSize > 0 ? pos - 1 : pos); }
    private static int right(int pos) { return (pos % squareSize < squareSize - 1 ? pos + 1 : pos); }
    private static int above(int pos) { return (pos >= squareSize ? pos - squareSize : pos); }
    private static int below(int pos) { return (pos < nbStates - squareSize ? pos + squareSize : pos); }

} // End of class FrozenLakeCPService