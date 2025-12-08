# TI-Nspire compatible Neural Network Weight Designer

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def to_fraction(decimal, max_denom=10000):
    if decimal == int(decimal):
        return "{}/1".format(int(decimal))
    
    sign = 1 if decimal >= 0 else -1
    decimal = abs(decimal)
    
    for denom in range(1, max_denom + 1):
        numer = round(decimal * denom)
        if abs(decimal - numer / denom) < 0.0001:
            g = gcd(numer, denom)
            numer //= g
            denom //= g
            if sign < 0:
                return "-{}/{}".format(numer, denom)
            return "{}/{}".format(numer, denom)
    
    return str(decimal)

def step_activation(z):
    return 1 if z > 0 else 0

def sigmoid_activation(z):
    import math
    if z < -20:
        return 0
    if z > 20:
        return 1
    return 1.0 / (1 + math.exp(-z))

def relu_activation(z):
    return max(0, z)

def apply_activation(z, act_type):
    if act_type == 1:
        return step_activation(z)
    elif act_type == 2:
        return sigmoid_activation(z)
    elif act_type == 3:
        return relu_activation(z)
    return z

def matrix_vector_mult(matrix, vector):
    result = []
    for row in matrix:
        val = sum(row[i] * vector[i] for i in range(len(vector)))
        result.append(val)
    return result

def generate_binary_combinations(n):
    # Generate all binary combinations of length n
    # Replaces itertools.product([0, 1], repeat=n)
    if n == 0:
        return [[]]
    
    result = []
    for i in range(2 ** n):
        combo = []
        num = i
        for j in range(n):
            combo.append(num % 2)
            num //= 2
        result.append(combo)
    return result

def test_truth_table(weights, activation_type):
    n_inputs = len(weights) - 1
    print("\nTruth table verification:")
    print("-" * 30)
    
    for combo in generate_binary_combinations(n_inputs):
        input_vec = [1] + combo
        z = sum(weights[i] * input_vec[i] for i in range(len(weights)))
        output = apply_activation(z, activation_type)
        combo_str = ", ".join("a{}={}".format(i+1, combo[i]) for i in range(n_inputs))
        print("{}: z={:.2f} -> output={}".format(combo_str, z, output))

def solve_threshold(var_idx, threshold, inequality, input_size):
    # Solve for weights such that sum > 0 iff variable > threshold (or <)
    # var_idx is the position in the input vector (e.g., 2 for q in [1,p,q])
    weights = [0] * input_size
    
    if inequality == '>':
        weights[0] = -threshold
        weights[var_idx] = 1
    else:  # '<'
        weights[0] = threshold
        weights[var_idx] = -1
    
    return weights

def find_logic_weights(truth_table, n_inputs, activation_type):
    # Try to find weights that satisfy the truth table
    # Simple search for step activation
    
    if activation_type != 1:
        print("Auto-search only works for step activation")
        return None
    
    # Generate all weight combinations manually (no itertools)
    # Search in order of increasing absolute values
    for max_val in range(1, 11):
        for bias in range(-max_val, max_val + 1):
            # Generate all combinations for weights
            weight_combos = generate_weight_combinations(n_inputs, max_val)
            
            for weights_tuple in weight_combos:
                weights = [bias] + weights_tuple
                
                valid = True
                has_edge_case = False
                
                for combo, expected in truth_table:
                    input_vec = [1] + list(combo)
                    z = sum(weights[i] * input_vec[i] for i in range(len(weights)))
                    output = step_activation(z)
                    
                    # Avoid solutions where z=0 (edge case)
                    if z == 0:
                        has_edge_case = True
                        break
                    
                    if output != expected:
                        valid = False
                        break
                
                if valid and not has_edge_case:
                    return weights
    
    return None

def generate_weight_combinations(n, max_val):
    # Generate all combinations of n weights in range [-max_val, max_val]
    # Replaces itertools.product(range(-max_val, max_val + 1), repeat=n)
    if n == 0:
        return [[]]
    
    result = []
    total_values = 2 * max_val + 1
    total_combos = total_values ** n
    
    for i in range(total_combos):
        combo = []
        num = i
        for j in range(n):
            val = (num % total_values) - max_val
            combo.append(val)
            num //= total_values
        result.append(combo)
    
    return result

# Main program starts here
print("=" * 20)
print("Neural Network Designer")
print("=" * 20)

# Define network architecture
num_layers = int(input("How many layers total? "))

layers = []

# Input layer
print("\n--- Layer 1 (Input) ---")
input_dim = int(input("Input dimension (with bias): "))
var_names_str = input("Variable names (comma-separated): ")
var_names = [v.strip() for v in var_names_str.split(',')]
layers.append({'size': input_dim, 'vars': var_names, 'bias_out': False})

# Hidden and output layers
for layer_idx in range(2, num_layers + 1):
    print("\n--- Layer {} {} ---".format(layer_idx, "(Output)" if layer_idx == num_layers else "(Hidden)"))
    n_neurons = int(input("Number of neurons: "))
    
    if layer_idx < num_layers:
        bias_out = input("Add bias to output? (y/n): ").strip().lower() == 'y'
    else:
        bias_out = False
    
    layers.append({'size': n_neurons, 'bias_out': bias_out})

# Activation function
print("\nActivation function:")
print("1) Step: g(z) = 1 if z>0, else 0")
print("2) Sigmoid: g(z) = 1/(1+exp(-z))")
print("3) ReLU: g(z) = max(0, z)")
activation_type = int(input("Choice: "))

# Display generated structure
print("\n" + "=" * 20)
print("Generated network structure:")
print("-" * 20)

for i in range(1, num_layers):
    src_idx = i - 1  # layer index in array (0-indexed)
    dst_idx = i  # layer index in array (0-indexed)
    
    if i == 1:
        src_symbol = "x"
    else:
        src_symbol = "a^({})".format(i)
    
    print("z^({}) = Theta^({}) * {}".format(i + 1, i, src_symbol))
    print("a^({}) = g(z^({}))".format(i + 1, i + 1))
    
    if i < num_layers - 1 and layers[dst_idx]['bias_out']:
        neuron_list = ", ".join("a^({})_{}".format(i + 1, j+1) for j in range(layers[dst_idx]['size']))
        print("a^({}) = [1, {}]^T".format(i + 1, neuron_list))
    elif i == num_layers - 1:
        if layers[dst_idx]['size'] == 1:
            print("h = a^({}) = g(z^({}))".format(i + 1, i + 1))
        else:
            print("h = a^({})".format(i + 1))
    print()

confirm = input("Confirm? (y/n): ").strip().lower()
if confirm != 'y':
    print("Exiting...")
    exit()

# Store weight matrices
weight_matrices = []
for i in range(num_layers - 1):
    weight_matrices.append(None)

# Main menu loop
while True:
    print("\n" + "=" * 20)
    print("Main Menu")
    print("=" * 20)
    print("1) Design layer weights")
    print("2) Test network with input")
    print("3) View all weights")
    print("4) Test boundary cases")
    print("q) Quit")
    
    choice = input("Choice: ").strip().lower()
    
    if choice == 'q':
        break
    
    elif choice == '1':
        # Design weights for a specific layer
        print("\nWhich layer to design?")
        for i in range(num_layers - 1):
            dst_idx = i + 2
            print("{}) Theta^({}) (Layer {} -> {})".format(i+1, i+1, i+1, dst_idx))
        
        layer_choice = int(input("Choice: ")) - 1
        
        if layer_choice < 0 or layer_choice >= num_layers - 1:
            print("Invalid choice")
            continue
        
        src_idx = layer_choice
        dst_idx = layer_choice + 1
        
        src_size = layers[src_idx]['size']
        if src_idx > 0 and layers[src_idx]['bias_out']:
            src_size = 1 + layers[src_idx]['size']
        
        dst_size = layers[dst_idx]['size']
        
        print("\n" + "=" * 20)
        print("Designing Theta^({}) ({}x{} matrix)".format(layer_choice + 1, dst_size, src_size))
        print("=" * 20)
        
        matrix = []
        
        for neuron_idx in range(dst_size):
            print("\n--- Neuron {} (a^({})_{}) ---".format(neuron_idx + 1, dst_idx + 1, neuron_idx + 1))
            print("Method:")
            print("1) Manual: Enter coefficients")
            print("2) Threshold: Variable > or < constant")
            print("3) Truth table: Specify outputs")
            
            method = int(input("Choice: "))
            
            if method == 1:
                print("Enter {} weights (comma-separated):".format(src_size))
                weights_str = input("Weights: ")
                weights = [float(x.strip()) for x in weights_str.split(',')]
                matrix.append(weights)
            
            elif method == 2:
                print("\nThreshold type:")
                print("a) Variable > constant")
                print("b) Variable < constant")
                threshold_type = input("Choice: ").strip().lower()
                
                print("\nAvailable variables:")
                for idx, var in enumerate(layers[src_idx]['vars']):
                    if var != '1':
                        print("{}) {}".format(idx, var))
                
                var_idx = int(input("Variable index: "))
                print("\nExample: For 'q > 4', enter: 4")
                constant = float(input("Constant value: "))
                
                inequality = '>' if threshold_type == 'a' else '<'
                weights = solve_threshold(var_idx, constant, inequality, src_size)
                
                var_name = layers[src_idx]['vars'][var_idx]
                print("\nSolving: {} {} {}".format(var_name, inequality, constant))
                weights_str = ", ".join(str(w) for w in weights)
                print("Theta^({})[{}] = [{}]".format(layer_choice + 1, neuron_idx + 1, weights_str))
                
                matrix.append(weights)
            
            elif method == 3:
                n_inputs = src_size - 1  # exclude bias
                
                print("\nTruth table for inputs (bias excluded):")
                truth_table = []
                
                for combo in generate_binary_combinations(n_inputs):
                    combo_str = ", ".join(str(c) for c in combo)
                    output = int(input("Input ({}): Output? ".format(combo_str)))
                    truth_table.append((combo, output))
                
                print("\nSearching for weights...")
                weights = find_logic_weights(truth_table, n_inputs, activation_type)
                
                if weights is None:
                    print("Could not find weights automatically.")
                    print("Enter manually (comma-separated):")
                    weights_str = input("Weights: ")
                    weights = [float(x.strip()) for x in weights_str.split(',')]
                else:
                    weights_str = ", ".join(str(w) for w in weights)
                    print("Found: [{}]".format(weights_str))
                    test_truth_table(weights, activation_type)
                
                matrix.append(weights)
        
        weight_matrices[layer_choice] = matrix
        print("\nTheta^({}) saved!".format(layer_choice + 1))
        input("Press Enter to continue...")
    
    elif choice == '2':
        # Test network
        if any(w is None for w in weight_matrices):
            print("\nNot all weight matrices defined!")
            continue
        
        print("\nEnter input values (comma-separated):")
        input_str = input("Input: ")
        current_activation = [float(x.strip()) for x in input_str.split(',')]
        
        print("\n" + "=" * 20)
        print("Forward Pass")
        print("=" * 20)
        
        for layer_idx in range(num_layers - 1):
            dst_idx = layer_idx + 1
            
            print("\n--- Layer {} -> {} ---".format(layer_idx + 1, dst_idx + 1))
            
            weights = weight_matrices[layer_idx]
            z_values = matrix_vector_mult(weights, current_activation)
            
            new_activation = []
            for neuron_idx, z in enumerate(z_values):
                a = apply_activation(z, activation_type)
                new_activation.append(a)
                print("z^({})_{} = {:.4f} -> a^({})_{} = {}".format(
                    dst_idx + 1, neuron_idx + 1, z, dst_idx + 1, neuron_idx + 1, 
                    to_fraction(a) if activation_type != 2 else "{:.4f}".format(a)))
            
            if layers[dst_idx]['bias_out']:
                current_activation = [1] + new_activation
                print("a^({}) = [1, {}]".format(dst_idx + 1, ", ".join(to_fraction(x) for x in new_activation)))
            else:
                current_activation = new_activation
        
        if layers[-1]['size'] == 1:
            print("\nFinal output h = {}".format(to_fraction(current_activation[0]) if activation_type != 2 else "{:.4f}".format(current_activation[0])))
        else:
            print("\nFinal output h = [{}]".format(", ".join(to_fraction(x) for x in current_activation)))
        
        input("\nPress Enter to continue...")
    
    elif choice == '3':
        # View all weights
        print("\n" + "=" * 20)
        print("All Weight Matrices")
        print("=" * 20)
        
        for layer_idx in range(num_layers - 1):
            print("\nTheta^({}):".format(layer_idx + 1))
            if weight_matrices[layer_idx] is None:
                print("  Not defined yet")
            else:
                for row_idx, row in enumerate(weight_matrices[layer_idx]):
                    row_str = ", ".join("{:6.2f}".format(x) for x in row)
                    print("  [{}]".format(row_str))
        
        input("\nPress Enter to continue...")
    
    elif choice == '4':
        # Test boundary cases
        if any(w is None for w in weight_matrices):
            print("\nNot all weight matrices defined!")
            continue
        
        print("\nTest all binary combinations for hidden layer?")
        if layers[0]['size'] > 5:
            print("Too many inputs (>5) for exhaustive testing")
            continue
        
        n_inputs = layers[0]['size'] - 1  # exclude bias
        
        print("\n" + "=" * 20)
        print("Boundary Testing")
        print("=" * 20)
        
        for combo in generate_binary_combinations(n_inputs):
            current_activation = [1] + combo
            combo_str = ", ".join("{}={}".format(layers[0]['vars'][i+1], combo[i]) for i in range(n_inputs))
            
            for layer_idx in range(num_layers - 1):
                weights = weight_matrices[layer_idx]
                z_values = matrix_vector_mult(weights, current_activation)
                new_activation = [apply_activation(z, activation_type) for z in z_values]
                
                if layers[layer_idx + 1]['bias_out']:
                    current_activation = [1] + new_activation
                else:
                    current_activation = new_activation
            
            output = current_activation[0] if layers[-1]['size'] == 1 else current_activation
            print("Input ({}): Output = {}".format(combo_str, output))
        
        input("\nPress Enter to continue...")
    
    else:
        print("Invalid choice")

print("\nGoodbye!")
