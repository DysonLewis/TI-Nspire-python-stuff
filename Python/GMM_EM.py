# TI-Nspire compatible GMM EM calculator

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

def gaussian(x, mu, sigma_sq):
    import math
    return (1.0 / math.sqrt(2 * math.pi * sigma_sq)) * math.exp(-((x - mu) ** 2) / (2 * sigma_sq))

def compute_m_step(x_values, gamma_n1):
    n = len(x_values)
    
    gamma_n2 = [1 - g for g in gamma_n1]
    
    sum_gamma_n1 = sum(gamma_n1)
    sum_gamma_n2 = sum(gamma_n2)
    
    omega = sum_gamma_n1 / n
    
    mu1 = sum(gamma_n1[i] * x_values[i] for i in range(n)) / sum_gamma_n1
    
    mu2 = sum(gamma_n2[i] * x_values[i] for i in range(n)) / sum_gamma_n2
    
    return omega, mu1, mu2

def compute_e_step(x_values, omega, mu1, mu2, sigma_sq):
    gamma_n1 = []
    for x in x_values:
        n1 = omega * gaussian(x, mu1, sigma_sq)
        n2 = (1 - omega) * gaussian(x, mu2, sigma_sq)
        denom = n1 + n2
        if denom > 0:
            gamma_n1.append(n1 / denom)
        else:
            gamma_n1.append(0.5)
    return gamma_n1

print("GMM EM Calculator (1D, 2 clusters)")
print("=" * 30)

n = int(input("Enter number of data points: "))
x_values = []
gamma_n1 = []

print("Enter x_n and gamma_n1 (comma-separated):")
for i in range(n):
    vals = list(map(float, input("Point {}: ".format(i + 1)).split(',')))
    x_values.append(vals[0])
    gamma_n1.append(vals[1])

print("\nData loaded successfully")

while True:
    print("\n" + "=" * 30)
    print("Select option:")
    print("a) M-step: compute omega, mu1, mu2")
    print("b) E-step: compute gamma values")
    print("c) Full EM iteration")
    print("r) Re-enter data")
    print("q) Quit")
    
    choice = input("Choice: ").strip().lower()
    
    if choice == 'q':
        break
    
    elif choice == 'r':
        n = int(input("Enter number of data points: "))
        x_values = []
        gamma_n1 = []
        
        print("Enter x_n and gamma_n1 (comma-separated):")
        for i in range(n):
            vals = list(map(float, input("Point {}: ".format(i + 1)).split(',')))
            x_values.append(vals[0])
            gamma_n1.append(vals[1])
        
        print("\nData reloaded successfully")
    
    elif choice == 'a':
        omega, mu1, mu2 = compute_m_step(x_values, gamma_n1)
        
        print("\nM-step results:")
        print("omega = {} or {:.4f}".format(to_fraction(omega), omega))
        print("mu1 = {} or {:.4f}".format(to_fraction(mu1), mu1))
        print("mu2 = {} or {:.4f}".format(to_fraction(mu2), mu2))
        input("Press 'm' for menu: ")
    
    elif choice == 'b':
        omega = float(input("Enter omega: "))
        mu1 = float(input("Enter mu1: "))
        mu2 = float(input("Enter mu2: "))
        sigma_sq = float(input("Enter sigma^2 (variance): "))
        
        gamma_n1 = compute_e_step(x_values, omega, mu1, mu2, sigma_sq)
        
        print("\nE-step results:")
        for i in range(n):
            print("gamma_{}1 = {} or {:.4f}".format(i+1, to_fraction(gamma_n1[i]), gamma_n1[i]))
        input("Press 'm' for menu: ")
    
    elif choice == 'c':
        omega = float(input("Initial omega: "))
        mu1 = float(input("Initial mu1: "))
        mu2 = float(input("Initial mu2: "))
        sigma_sq = float(input("Enter sigma^2 (variance): "))
        
        max_iter = int(input("Max iterations: "))
        
        for iteration in range(max_iter):
            print("\nIteration {}:".format(iteration + 1))
            print("Parameters:")
            print("omega = {} or {:.4f}".format(to_fraction(omega), omega))
            print("mu1 = {} or {:.4f}".format(to_fraction(mu1), mu1))
            print("mu2 = {} or {:.4f}".format(to_fraction(mu2), mu2))
            
            gamma_n1 = compute_e_step(x_values, omega, mu1, mu2, sigma_sq)
            
            print("\nE-step (gamma values):")
            for i in range(n):
                print("gamma_{}1 = {} or {:.4f}".format(i+1, to_fraction(gamma_n1[i]), gamma_n1[i]))
            
            omega, mu1, mu2 = compute_m_step(x_values, gamma_n1)
        
        print("\nFinal parameters:")
        print("omega = {} or {:.4f}".format(to_fraction(omega), omega))
        print("mu1 = {} or {:.4f}".format(to_fraction(mu1), mu1))
        print("mu2 = {} or {:.4f}".format(to_fraction(mu2), mu2))
        input("Press 'm' for menu: ")
    
    else:
        print("Invalid choice")
