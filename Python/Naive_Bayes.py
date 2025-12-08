# TI-Nspire compatible Naive Bayes calculator

def get_dataset(num_features):
    n = int(input("Enter number of data points: "))
    X = []
    Y = []
    print("Enter X1,...,X{}, Y (comma-separated) for each point:".format(num_features))
    for i in range(n):
        vals = list(map(int, input("Point {}: ".format(i + 1)).split(',')))
        X.append(vals[:-1])
        Y.append(vals[-1])
    return X, Y

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

def mle_all_single_params(X, Y, num_features):
    n = len(Y)
    y1_count = sum(1 for y in Y if y == 1)
    y0_count = n - y1_count
    
    results = {}
    
    for feat_idx in range(num_features):
        if y1_count > 0:
            results['P(X{}=0|Y=1)'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 1 and X[i][feat_idx] == 0) / y1_count
            results['P(X{}=1|Y=1)'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 1 and X[i][feat_idx] == 1) / y1_count
        else:
            results['P(X{}=0|Y=1)'.format(feat_idx+1)] = 0
            results['P(X{}=1|Y=1)'.format(feat_idx+1)] = 0
        
        if y0_count > 0:
            results['P(X{}=0|Y=0)'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 0 and X[i][feat_idx] == 0) / y0_count
            results['P(X{}=1|Y=0)'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 0 and X[i][feat_idx] == 1) / y0_count
        else:
            results['P(X{}=0|Y=0)'.format(feat_idx+1)] = 0
            results['P(X{}=1|Y=0)'.format(feat_idx+1)] = 0
    
    return results

def map_parameter(X, Y, feature, feature_val, y_val, prior_type, alpha=0, beta=0):
    n = len(Y)
    y_count = sum(1 for y in Y if y == y_val)
    
    if y_count == 0:
        return 0
    
    count = sum(1 for i in range(n) if Y[i] == y_val and X[i][feature-1] == feature_val)
    
    if prior_type == 1:
        a = count / (count + 2)
    elif prior_type == 2:
        a = (count + 1) / (y_count + 2)
    elif prior_type == 3:
        denom = y_count + alpha + beta - 2
        if denom <= 0:
            return count / y_count
        a = (count + alpha - 1) / denom
    else:
        a = count / y_count
    
    return a

def compute_all_mle(X, Y, num_features):
    n = len(Y)
    y1_count = sum(1 for y in Y if y == 1)
    y0_count = n - y1_count
    
    p_y1 = y1_count / n if n > 0 else 0
    
    params = {'p_y1': p_y1}
    
    for feat_idx in range(num_features):
        if y1_count > 0:
            params['p_x{}_y1'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 1 and X[i][feat_idx] == 1) / y1_count
        else:
            params['p_x{}_y1'.format(feat_idx+1)] = 0
        
        if y0_count > 0:
            params['p_x{}_y0'.format(feat_idx+1)] = sum(1 for i in range(n) if Y[i] == 0 and X[i][feat_idx] == 1) / y0_count
        else:
            params['p_x{}_y0'.format(feat_idx+1)] = 0
    
    return params

def compute_posterior(x, params, num_features):
    p_y1 = params['p_y1']
    p_y0 = 1 - p_y1
    
    prob_y1 = p_y1
    prob_y0 = p_y0
    
    for feat_idx in range(num_features):
        if x[feat_idx] == 1:
            prob_y1 *= params['p_x{}_y1'.format(feat_idx+1)]
            prob_y0 *= params['p_x{}_y0'.format(feat_idx+1)]
        else:
            prob_y1 *= (1 - params['p_x{}_y1'.format(feat_idx+1)])
            prob_y0 *= (1 - params['p_x{}_y0'.format(feat_idx+1)])
    
    denominator = prob_y1 + prob_y0
    
    if denominator == 0:
        return 0
    return prob_y1 / denominator

def em_m_step(X, Y, posteriors, num_features):
    n = len(X)
    
    sum_post = sum(posteriors)
    p_y1 = sum_post / n
    
    params = {'p_y1': p_y1}
    
    for feat_idx in range(num_features):
        num_x_y1 = sum(X[i][feat_idx] * posteriors[i] for i in range(n))
        p_x_y1 = num_x_y1 / sum_post if sum_post > 0 else 0
        params['p_x{}_y1'.format(feat_idx+1)] = p_x_y1
        
        sum_post_y0 = n - sum_post
        num_x_y0 = sum(X[i][feat_idx] * (1 - posteriors[i]) for i in range(n))
        p_x_y0 = num_x_y0 / sum_post_y0 if sum_post_y0 > 0 else 0
        params['p_x{}_y0'.format(feat_idx+1)] = p_x_y0
    
    return params

print("Naive Bayes Calculator")
print("=" * 30)

num_features = int(input("Enter number of features: "))
X, Y = get_dataset(num_features)
print("\nDataset loaded successfully")

while True:
    print("\n" + "=" * 30)
    print("Select option:")
    print("a) MLE for all conditionals")
    print("b) MAP for specific parameter")
    print("c) Compute all MLE parameters")
    print("d) Compute posterior probabilities")
    print("e) EM M-step: update from posteriors")
    print("f) Full EM iteration")
    print("r) Re-enter dataset")
    print("q) Quit")
    
    choice = input("Choice: ").strip().lower()
    
    if choice == 'q':
        break
    
    elif choice == 'r':
        num_features = int(input("Enter number of features: "))
        X, Y = get_dataset(num_features)
        print("\nDataset reloaded successfully")
    
    elif choice == 'a':
        results = mle_all_single_params(X, Y, num_features)
        print("\nMLE estimates for all conditional probabilities:")
        for key in sorted(results.keys()):
            print("{} = {} or {:.4f}".format(key, to_fraction(results[key]), results[key]))
        input("Press 'm' for menu: ")
    
    elif choice == 'b':
        print("\nWhich parameter to compute MAP for?")
        feature = int(input("Feature (1 to {}): ".format(num_features)))
        feature_val = int(input("Feature value (0 or 1): "))
        y_val = int(input("Given Y value (0 or 1): "))
        print("\nPrior distribution type:")
        print("1) P(a) = 3(1-a)^2")
        print("2) P(a) = 2(1-a)")
        print("3) Beta(alpha, beta) - Custom")
        print("4) Uniform (MLE)")
        prior_type = int(input("Choice: "))
        
        if prior_type == 3:
            print("\nBeta prior format: P(a) proportional to a^(alpha-1) * (1-a)^(beta-1)")
            alpha = float(input("Enter alpha: "))
            beta = float(input("Enter beta: "))
            
            def beta_function(a, b):
                if a == 1 and b == 1:
                    return 1.0
                product = 1.0
                for i in range(1, int(a)):
                    product *= i
                for i in range(1, int(b)):
                    product *= i
                for i in range(1, int(a + b)):
                    product /= i
                return product
            
            norm_const = 1.0 / beta_function(alpha, beta)
            
            exp_a = alpha - 1
            exp_b = beta - 1
            
            if exp_a == 0 and exp_b == 0:
                print("Using prior: P(a) = {}".format(round(norm_const, 4)))
            elif exp_a == 0:
                if exp_b == 1:
                    print("Using prior: P(a) = {}(1-a)".format(round(norm_const, 4)))
                else:
                    print("Using prior: P(a) = {}(1-a)^{}".format(round(norm_const, 4), exp_b))
            elif exp_b == 0:
                if exp_a == 1:
                    print("Using prior: P(a) = {}a".format(round(norm_const, 4)))
                else:
                    print("Using prior: P(a) = {}a^{}".format(round(norm_const, 4), exp_a))
            else:
                a_part = "a" if exp_a == 1 else "a^{}".format(exp_a)
                b_part = "(1-a)" if exp_b == 1 else "(1-a)^{}".format(exp_b)
                print("Using prior: P(a) = {}{}*{}".format(round(norm_const, 4), a_part, b_part))
            
            a = map_parameter(X, Y, feature, feature_val, y_val, prior_type, alpha, beta)
        elif prior_type == 1:
            print("Using prior: P(a) = 3(1-a)^2")
            a = map_parameter(X, Y, feature, feature_val, y_val, prior_type)
        elif prior_type == 2:
            print("Using prior: P(a) = 2(1-a)")
            a = map_parameter(X, Y, feature, feature_val, y_val, prior_type)
        else:
            print("Using uniform prior (MLE)")
            a = map_parameter(X, Y, feature, feature_val, y_val, prior_type)
        
        print("\nMAP estimate: P(X{}={}|Y={}) = {} or {:.4f}".format(feature, feature_val, y_val, to_fraction(a), a))
        input("Press 'm' for menu: ")
    
    elif choice == 'c':
        params = compute_all_mle(X, Y, num_features)
        print("\nMLE parameters:")
        print("P(Y=1) = {} or {:.4f}".format(to_fraction(params['p_y1']), params['p_y1']))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=1) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y1'.format(feat_idx+1)]), params['p_x{}_y1'.format(feat_idx+1)]))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=0) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y0'.format(feat_idx+1)]), params['p_x{}_y0'.format(feat_idx+1)]))
        input("Press 'm' for menu: ")
    
    elif choice == 'd':
        params = compute_all_mle(X, Y, num_features)
        print("\nParameters computed:")
        print("P(Y=1) = {} or {:.4f}".format(to_fraction(params['p_y1']), params['p_y1']))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=1) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y1'.format(feat_idx+1)]), params['p_x{}_y1'.format(feat_idx+1)]))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=0) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y0'.format(feat_idx+1)]), params['p_x{}_y0'.format(feat_idx+1)]))
        
        print("\nPosteriors for all combinations:")
        from itertools import product
        for combo in product([0, 1], repeat=num_features):
            post = compute_posterior(list(combo), params, num_features)
            combo_str = ",".join("X{}={}".format(i+1, combo[i]) for i in range(num_features))
            print("P(Y=1|{}) = {} or {:.4f}".format(combo_str, to_fraction(post), post))
        input("Press 'm' for menu: ")
    
    elif choice == 'e':
        n = len(X)
        posteriors = []
        print("\nEnter P(Y=1|X) for each point:")
        for i in range(n):
            x_str = ",".join(str(X[i][j]) for j in range(num_features))
            post = float(input("P(Y=1|X={}): ".format(x_str)))
            posteriors.append(post)
        
        new_params = em_m_step(X, Y, posteriors, num_features)
        print("\nUpdated parameters (M-step):")
        print("P(Y=1) = {} or {:.4f}".format(to_fraction(new_params['p_y1']), new_params['p_y1']))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=1) = {} or {:.4f}".format(feat_idx+1, to_fraction(new_params['p_x{}_y1'.format(feat_idx+1)]), new_params['p_x{}_y1'.format(feat_idx+1)]))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=0) = {} or {:.4f}".format(feat_idx+1, to_fraction(new_params['p_x{}_y0'.format(feat_idx+1)]), new_params['p_x{}_y0'.format(feat_idx+1)]))
        input("Press 'm' for menu: ")
    
    elif choice == 'f':
        params = compute_all_mle(X, Y, num_features)
        
        max_iter = int(input("Max iterations: "))
        for iteration in range(max_iter):
            print("\nIteration {}:".format(iteration + 1))
            print("Parameters:")
            print("P(Y=1) = {} or {:.4f}".format(to_fraction(params['p_y1']), params['p_y1']))
            for feat_idx in range(num_features):
                print("P(X{}=1|Y=1) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y1'.format(feat_idx+1)]), params['p_x{}_y1'.format(feat_idx+1)]))
            for feat_idx in range(num_features):
                print("P(X{}=1|Y=0) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y0'.format(feat_idx+1)]), params['p_x{}_y0'.format(feat_idx+1)]))
            
            posteriors = []
            print("\nPosteriors:")
            for i in range(len(X)):
                post = compute_posterior(X[i], params, num_features)
                posteriors.append(post)
                x_str = ",".join(str(X[i][j]) for j in range(num_features))
                print("P(Y=1|X={}) = {} or {:.4f}".format(x_str, to_fraction(post), post))
            
            params = em_m_step(X, Y, posteriors, num_features)
        
        print("\nFinal parameters:")
        print("P(Y=1) = {} or {:.4f}".format(to_fraction(params['p_y1']), params['p_y1']))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=1) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y1'.format(feat_idx+1)]), params['p_x{}_y1'.format(feat_idx+1)]))
        for feat_idx in range(num_features):
            print("P(X{}=1|Y=0) = {} or {:.4f}".format(feat_idx+1, to_fraction(params['p_x{}_y0'.format(feat_idx+1)]), params['p_x{}_y0'.format(feat_idx+1)]))
        input("Press 'm' for menu: ")
    
    else:
        print("Invalid choice")
