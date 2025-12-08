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

def dot_product(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def vector_subtract(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def scalar_mult(scalar, v):
    return [scalar * x for x in v]

def vector_norm(v):
    import math
    return math.sqrt(sum(x * x for x in v))

def get_dataset(n_features):
    n_examples = int(input("Enter number of examples: "))
    X = []
    Y = []
    
    print("Enter features and label:")
    for i in range(n_examples):
        vals_str = input("Ex {} (f1,f2,...,y): ".format(i + 1))
        vals = [float(x.strip()) for x in vals_str.split(',')]
        X.append(vals[:-1])
        Y.append(int(vals[-1]))
    
    return X, Y

def find_support_vectors_b0(X, Y):
    # Find support vectors for hard SVM with b=0
    # Support vectors are the closest points from opposite classes
    
    pos_indices = [i for i in range(len(Y)) if Y[i] == 1]
    neg_indices = [i for i in range(len(Y)) if Y[i] == -1]
    
    print("\nPositive examples:")
    for i in pos_indices:
        print("x_{}: {}".format(i+1, X[i]))
    
    print("\nNegative examples:")
    for i in neg_indices:
        print("x_{}: {}".format(i+1, X[i]))
    
    print("\nSelect support vectors:")
    print("Enter indices of support vectors (comma-separated):")
    sv_str = input("Support vector indices: ")
    sv_indices = [int(x.strip()) - 1 for x in sv_str.split(',')]
    
    return sv_indices

def auto_find_support_vectors_b0(X, Y):
    # Automatically find optimal support vectors for b=0
    
    pos_indices = [i for i in range(len(Y)) if Y[i] == 1]
    neg_indices = [i for i in range(len(Y)) if Y[i] == -1]
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print("Need both classes")
        return None
    
    best_margin = 0
    best_pair = None
    all_pairs = []
    
    print("Finding optimal margin...")
    print("Testing all pairs...")
    
    for i in pos_indices:
        for j in neg_indices:
            diff = vector_subtract(X[i], X[j])
            diff_norm_sq = dot_product(diff, diff)
            
            if diff_norm_sq < 0.0001:
                continue
            
            w_test = scalar_mult(2.0 / diff_norm_sq, diff)
            
            # Check constraints and find actual support vectors
            valid = True
            support_vecs = []
            min_constraint = float('inf')
            
            for k in range(len(X)):
                score = dot_product(w_test, X[k])
                constraint = Y[k] * score
                
                if constraint < min_constraint:
                    min_constraint = constraint
                
                # FIXED: detect support vectors cleanly
                if abs(constraint - 1.0) < 0.001:
                    support_vecs.append(k)
                
                # FIXED: enforce SVM constraint y(w·x) ≥ 1
                if constraint < 1.0:
                    valid = False
            
            if valid:
                w_norm_sq = dot_product(w_test, w_test)
                import math
                w_norm = math.sqrt(w_norm_sq)
                margin = 1.0 / w_norm
                
                sv_str = ",".join(["x_{}".format(s+1) for s in support_vecs])
                print("Pair x_{},x_{}: margin={:.4f} SVs:[{}]".format(
                    i+1, j+1, margin, sv_str))
                
                all_pairs.append((i, j, margin, support_vecs))
                
                if margin > best_margin:
                    best_margin = margin
                    best_pair = (i, j, support_vecs)
    
    if best_pair is None:
        print("No valid pair found")
        return None
    
    print("\nOptimal configuration:")
    print("Tested pair: x_{}, x_{}".format(best_pair[0]+1, best_pair[1]+1))
    print("Actual support vectors: {}".format(
        [i+1 for i in best_pair[2]]))
    print("Margin: {:.6f}".format(best_margin))
    
    return best_pair[2]

def compute_w_b0(X, Y, sv_indices):
    # Compute w for b=0 case
    # Using the constraint: w·x_pos = 1, w·x_neg = -1
    
    if len(sv_indices) != 2:
        print("Need exactly 2 support vectors")
        return None
    
    idx1, idx2 = sv_indices
    x1, y1 = X[idx1], Y[idx1]
    x2, y2 = X[idx2], Y[idx2]
    
    # Ensure x1 is positive, x2 is negative
    if y1 == -1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    
    print("\nPositive SV: x_{} = {}".format(idx1+1, x1))
    print("Negative SV: x_{} = {}".format(idx2+1, x2))
    
    # Solve: w·x1 = 1, w·x2 = -1
    # This gives: w·(x1-x2) = 2
    # Minimal norm solution: w = 2(x1-x2) / ||x1-x2||^2
    
    diff = vector_subtract(x1, x2)
    diff_norm_sq = dot_product(diff, diff)
    
    w = scalar_mult(2.0 / diff_norm_sq, diff)
    
    print("\nDifference x1 - x2 = {}".format(diff))
    print("||x1 - x2||^2 = {}".format(diff_norm_sq))
    
    return w

def compute_w_with_b(X, Y, sv_indices):
    # Compute w and b when b is allowed
    # Still uses: w·x1 + b = 1, w·x2 + b = -1
    
    if len(sv_indices) != 2:
        print("Need exactly 2 support vectors")
        return None, None
    
    idx1, idx2 = sv_indices
    x1, y1 = X[idx1], Y[idx1]
    x2, y2 = X[idx2], Y[idx2]
    
    # Ensure x1 is positive, x2 is negative
    if y1 == -1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    
    print("\nPositive SV: x_{} = {}".format(idx1+1, x1))
    print("Negative SV: x_{} = {}".format(idx2+1, x2))
    
    # From w·x1 + b = 1 and w·x2 + b = -1
    # We get: w·(x1-x2) = 2
    # So w is parallel to (x1-x2)
    
    diff = vector_subtract(x1, x2)
    diff_norm_sq = dot_product(diff, diff)
    
    w = scalar_mult(2.0 / diff_norm_sq, diff)
    
    # Now compute b from: w·x1 + b = 1
    b = 1.0 - dot_product(w, x1)
    
    print("\nDifference x1 - x2 = {}".format(diff))
    print("||x1 - x2||^2 = {}".format(diff_norm_sq))
    
    return w, b

def compute_margin(w):
    import math
    # ||w|| = sqrt(w1^2 + w2^2 + ...)
    w_norm_sq = sum(x * x for x in w)
    w_norm = math.sqrt(w_norm_sq)
    margin = 1.0 / w_norm
    
    # Try to express w_norm_sq as a fraction
    frac_str = to_fraction(w_norm_sq)
    
    if "/" in frac_str:
        parts = frac_str.split("/")
        num = int(parts[0]) if parts[0][0] != '-' else int(parts[0][1:])
        den = int(parts[1])
        w_norm_str = "sqrt({})/sqrt({})".format(num, den)
        margin_str = "sqrt({})/sqrt({})".format(den, num)
    else:
        val = int(frac_str)
        # Check if perfect square
        sqrt_val = int(math.sqrt(val) + 0.5)
        if abs(sqrt_val * sqrt_val - val) < 0.0001:
            w_norm_str = str(sqrt_val)
            margin_str = "1/{}".format(sqrt_val)
        else:
            w_norm_str = "sqrt({})".format(val)
            margin_str = "sqrt({})/{}".format(val, val)
    
    return w_norm, margin, w_norm_str, margin_str

def verify_constraints(X, Y, w, b=0):
    print("Verifying constraints:")
    print("-" * 20)
    
    all_satisfied = True
    for i in range(len(X)):
        score = dot_product(w, X[i]) + b
        constraint = Y[i] * score
        status = "OK" if constraint >= 0.99 else "BAD"
        
        if constraint < 0.99:
            all_satisfied = False
        
        print("x_{}: y*val={:.3f} {}".format(i+1, constraint, status))
    
    input("Press Enter...")
    return all_satisfied

def polynomial_kernel(x1, x2, degree):
    # K(x1, x2) = (x1^T x2)^degree
    return dot_product(x1, x2) ** degree

def compute_kernel_matrix(X, degree):
    n = len(X)
    K = []
    for i in range(n):
        row = []
        for j in range(n):
            k_val = polynomial_kernel(X[i], X[j], degree)
            row.append(k_val)
        K.append(row)
    return K

# Main program
print("=" * 20)
print("SVM Solver")
print("=" * 20)

n_features = int(input("Number of features: "))
X, Y = get_dataset(n_features)

print("Dataset loaded:")
for i in range(len(X)):
    print("x_{}: {}, y={}".format(i+1, X[i], Y[i]))
input("Press Enter...")

while True:
    print("\n" + "=" * 20)
    print("Main Menu")
    print("=" * 20)
    print("1) Find support vectors")
    print("1a) Auto find optimal SVs")
    print("2) Compute w* (b=0)")
    print("3) Compute w*, b* (b!=0)")
    print("4) Verify constraints")
    print("5) Kernel computations")
    print("6) View dataset")
    print("r) Re-enter dataset")
    print("q) Quit")
    
    choice = input("Choice: ").strip().lower()
    
    if choice == 'q':
        break
    
    elif choice == 'r':
        n_features = int(input("Number of features: "))
        X, Y = get_dataset(n_features)
        print("\nDataset reloaded")
    
    elif choice == '1':
        sv_indices = find_support_vectors_b0(X, Y)
        print("Support vectors:")
        print([i+1 for i in sv_indices])
        input("Press Enter...")
    
    elif choice == '1a':
        sv_indices = auto_find_support_vectors_b0(X, Y)
        if sv_indices:
            print("Support vectors:")
            print([i+1 for i in sv_indices])
        input("Press Enter...")
    
    elif choice == '2':
        print("--- Compute w* (b=0) ---")
        sv_str = input("SV indices (ex: 1,4): ")
        sv_indices = [int(x.strip()) - 1 for x in sv_str.split(',')]
        
        w = compute_w_b0(X, Y, sv_indices)
        
        if w:
            w_norm, margin, w_norm_str, margin_str = compute_margin(w)
            
            print("=" * 20)
            print("Results (b=0)")
            print("=" * 20)
            print("w* = {}".format([to_fraction(x) for x in w]))
            print("w* = {}".format(["%.6f" % x for x in w]))
            print("||w*|| = {}".format(w_norm_str))
            print("||w*|| = {:.6f}".format(w_norm))
            print("Margin = {}".format(margin_str))
            print("Margin = {:.6f}".format(margin))
            
            verify_constraints(X, Y, w, 0)
        
        input("Press Enter...")
    
    elif choice == '3':
        print("--- Compute w*, b* ---")
        sv_str = input("SV indices (ex: 1,4): ")
        sv_indices = [int(x.strip()) - 1 for x in sv_str.split(',')]
        
        w, b = compute_w_with_b(X, Y, sv_indices)
        
        if w:
            w_norm, margin, w_norm_str, margin_str = compute_margin(w)
            
            print("=" * 20)
            print("Results (b!=0)")
            print("=" * 20)
            print("w* = {}".format([to_fraction(x) for x in w]))
            print("w* = {}".format(["%.6f" % x for x in w]))
            print("b* = {}".format(to_fraction(b)))
            print("b* = {:.6f}".format(b))
            print("||w*|| = {}".format(w_norm_str))
            print("||w*|| = {:.6f}".format(w_norm))
            print("Margin = {}".format(margin_str))
            print("Margin = {:.6f}".format(margin))
            
            verify_constraints(X, Y, w, b)
        
        input("Press Enter...")
    
    elif choice == '4':
        print("--- Verify Constraints ---")
        
        print("Enter w (comma-separated):")
        w_str = input("w: ")
        w = [float(x.strip()) for x in w_str.split(',')]
        
        b_str = input("Enter b: ")
        b = float(b_str) if b_str.strip() else 0
        
        verify_constraints(X, Y, w, b)
    
    elif choice == '5':
        print("--- Kernels ---")
        print("1) K(xi, xj)")
        print("2) Kernel matrix")
        
        sub_choice = input("Choice: ").strip()
        
        if sub_choice == '1':
            degree = int(input("Degree: "))
            i = int(input("Index i: ")) - 1
            j = int(input("Index j: ")) - 1
            
            k_val = polynomial_kernel(X[i], X[j], degree)
            
            print("K(x_{},x_{})=(x_i^T x_j)^{}".format(i+1, j+1, degree))
            print("x_{} = {}".format(i+1, X[i]))
            print("x_{} = {}".format(j+1, X[j]))
            dot_val = dot_product(X[i], X[j])
            print("x_i^T x_j = {}".format(dot_val))
            print("K = {}".format(k_val))
            input("Press Enter...")
        
        elif sub_choice == '2':
            degree = int(input("Degree: "))
            K = compute_kernel_matrix(X, degree)
            
            print("Kernel matrix (deg {})".format(degree))
            print("-" * 20)
            for i in range(len(K)):
                row_str = " ".join("{:6.1f}".format(K[i][j]) for j in range(len(K[i])))
                print(row_str)
        
        input("Press Enter...")
    
    elif choice == '6':
        print("=" * 20)
        print("Dataset")
        print("=" * 20)
        for i in range(len(X)):
            print("x_{}: {}, y={}".format(i+1, X[i], Y[i]))
        input("Press Enter...")
    
    else:
        print("Invalid choice")

print("\nGoodbye!")

