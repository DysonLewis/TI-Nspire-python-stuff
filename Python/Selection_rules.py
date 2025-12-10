# TI-Nspire compatible Hydrogen Atom Selection Rules Calculator

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

def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def clebsch_gordan(l, ml, lp, q, lpp, mlpp):
    if mlpp != ml + q:
        return 0
    if abs(l - lp) > lpp or lpp > l + lp:
        return 0
    if abs(ml) > l or abs(q) > lp or abs(mlpp) > lpp:
        return 0
    
    if l == 1 and lp == 1:
        if lpp == 0:
            if ml == 1 and q == -1:
                return 1.0 / (3**0.5)
            elif ml == 0 and q == 0:
                return 1.0 / (3**0.5)
            elif ml == -1 and q == 1:
                return 1.0 / (3**0.5)
        elif lpp == 2:
            if ml == 1 and q == 1:
                return 1.0
            elif ml == 1 and q == 0:
                return 0.5
            elif ml == 1 and q == -1:
                return (1.0/6)**0.5
            elif ml == 0 and q == 1:
                return 0.5
            elif ml == 0 and q == 0:
                return 0
            elif ml == 0 and q == -1:
                return -0.5
    
    if l == 1 and lp == 1 and lpp == 1:
        if ml == 1 and q == 0:
            return (0.5)**0.5
        elif ml == 1 and q == -1:
            return -(0.5)**0.5
        elif ml == 0 and q == 1:
            return -(0.5)**0.5
        elif ml == 0 and q == -1:
            return (0.5)**0.5
        elif ml == -1 and q == 1:
            return (0.5)**0.5
        elif ml == -1 and q == 0:
            return -(0.5)**0.5
    
    return 0

def compute_expectation_values(n, l, ml, ms):
    E = -13.6 / (n**2)
    
    L2 = l * (l + 1)
    
    Lz = ml
    
    S2 = 0.75
    
    Sz = ms
    
    Sx_exp = 0
    Sy_exp = 0
    Sz_exp = ms
    Lx_exp = 0
    Ly_exp = 0
    Lz_exp = ml
    
    Sx2_exp = 0.25
    Sy2_exp = 0.25
    Sz2_exp = ms**2
    Lx2_exp = 0.5 * (l * (l + 1) - ml**2)
    Ly2_exp = 0.5 * (l * (l + 1) - ml**2)
    Lz2_exp = ml**2
    
    var_Sx = Sx2_exp - Sx_exp**2
    var_Sy = Sy2_exp - Sy_exp**2
    var_Sz = Sz2_exp - Sz_exp**2
    var_Lx = Lx2_exp - Lx_exp**2
    var_Ly = Ly2_exp - Ly_exp**2
    var_Lz = Lz2_exp - Lz_exp**2
    
    Jz = Lz + Sz
    J2_vals = []
    for j_val_x2 in range(abs(2*l - 1), 2*l + 2, 2):
        j_val = j_val_x2 / 2.0
        J2_vals.append(j_val * (j_val + 1))
    
    return {
        'H': E, 'L2': L2, 'Lz': Lz, 'S2': S2, 'Sz': Sz,
        'Sx': Sx_exp, 'Sy': Sy_exp, 'Sz': Sz_exp, 'Sx2': Sx2_exp, 'Sy2': Sy2_exp, 'Sz2': Sz2_exp,
        'var_Sx': var_Sx, 'var_Sy': var_Sy, 'var_Sz': var_Sz,
        'Lx': Lx_exp, 'Ly': Ly_exp, 'Lz': Lz_exp, 'Lx2': Lx2_exp, 'Ly2': Ly2_exp, 'Lz2': Lz2_exp,
        'var_Lx': var_Lx, 'var_Ly': var_Ly, 'var_Lz': var_Lz,
        'Jz': Jz, 'J2_possible': J2_vals
    }

def check_dipole_selection(n, l, ml, ms, np, lp, mlp, msp, E_dirs):
    reasons = []
    allowed = False
    allowed_dirs = []
    
    if msp != ms:
        reasons.append("Spin must be conserved")
        return False, reasons, []
    
    delta_l = lp - l
    if delta_l not in [-1, 1]:
        reasons.append("Delta_l must be +/-1")
        return False, reasons, []
    
    for E_dir in E_dirs:
        dir_allowed = False
        if E_dir == 'z':
            if mlp == ml:
                dir_allowed = True
                allowed_dirs.append(('z', 0))
        else:
            if mlp in [ml - 1, ml + 1]:
                dir_allowed = True
                q = mlp - ml
                allowed_dirs.append((E_dir, q))
        
        if dir_allowed:
            allowed = True
    
    if not allowed:
        delta_ml = mlp - ml
        reasons.append("Delta_ml={} not allowed".format(delta_ml))
        reasons.append("for E directions: {}".format(E_dirs))
        return False, reasons, []
    else:
        return True, ["Allowed by selection rules"], allowed_dirs

def format_frac(val):
    if abs(val) < 0.0001:
        return "0"
    if val == int(val):
        return str(int(val))
    if abs(val - 0.5) < 0.01:
        return "1/2"
    if abs(val + 0.5) < 0.01:
        return "-1/2"
    return to_fraction(val)

def print_header(text):
    print("=" * 30)
    print(text)
    print("=" * 30)

print("Hydrogen Selection Rules Calc")
print("=" * 30)

while True:
    print("Enter initial state:")
    n = int(input("n: ").strip())
    l = int(input("l: ").strip())
    ml = int(input("ml: ").strip())
    ms_input = input("ms (+ or -): ").strip()
    ms = 0.5 if '+' in ms_input else -0.5
    
    print("\n")
    print_header("Initial State")
    print("|n={}, l={}, ml={}, ms={}>".format(n, l, ml, format_frac(ms)))
    input("Press Enter...")
    
    exp_vals = compute_expectation_values(n, l, ml, ms)
    
    print("\n")
    print_header("Expectation Values (1/3)")
    print("<H> = -13.6/n^2 eV = {:.3f} eV".format(exp_vals['H']))
    print("<L^2> = h_b^2 l(l+1) = {} h_b^2".format(format_frac(exp_vals['L2'])))
    print("<Lz> = h_b ml = {} h_b".format(format_frac(exp_vals['Lz'])))
    print("<Lx> = {} h_b".format(format_frac(exp_vals['Lx'])))
    print("<Ly> = {} h_b".format(format_frac(exp_vals['Ly'])))
    print("<Lx^2> = {} h_b^2".format(format_frac(exp_vals['Lx2'])))
    print("<Ly^2> = {} h_b^2".format(format_frac(exp_vals['Ly2'])))
    print("<Lz^2> = {} h_b^2".format(format_frac(exp_vals['Lz2'])))
    input("Press Enter...")
    
    print("\n")
    print_header("Expectation Values (2/3)")
    print("Var(Lx) = {} h_b^2".format(format_frac(exp_vals['var_Lx'])))
    print("Var(Ly) = {} h_b^2".format(format_frac(exp_vals['var_Ly'])))
    print("Var(Lz) = {} h_b^2".format(format_frac(exp_vals['var_Lz'])))
    print("<S^2> = h_b^2 s(s+1) = {} h_b^2".format(format_frac(exp_vals['S2'])))
    print("<Sz> = h_b ms = {} h_b".format(format_frac(exp_vals['Sz'])))
    print("<Sx> = {} h_b".format(format_frac(exp_vals['Sx'])))
    print("<Sy> = {} h_b".format(format_frac(exp_vals['Sy'])))
    input("Press Enter...")
    
    print("\n")
    print_header("Expectation Values (3/3)")
    print("<Sx^2> = {} h_b^2".format(format_frac(exp_vals['Sx2'])))
    print("<Sy^2> = {} h_b^2".format(format_frac(exp_vals['Sy2'])))
    print("<Sz^2> = {} h_b^2".format(format_frac(exp_vals['Sz2'])))
    print("Var(Sx) = {} h_b^2".format(format_frac(exp_vals['var_Sx'])))
    print("Var(Sy) = {} h_b^2".format(format_frac(exp_vals['var_Sy'])))
    print("Var(Sz) = {} h_b^2".format(format_frac(exp_vals['var_Sz'])))
    print("<Jz> = <Lz>+<Sz> = {} h_b".format(format_frac(exp_vals['Jz'])))
    print("<J^2> can be: {} h_b^2".format([format_frac(x) for x in exp_vals['J2_possible']]))
    input("Press Enter...")
    
    print("\n")
    print_header("Dipole Interaction")
    print("E field form:")
    print("Examples:")
    print("  z  (E0*z)")
    print("  x  (E0*x)")
    print("  2*x+3*z  (mixture)")
    print("  x*y  (higher order)")
    E_input = input("E field: ").strip().lower()
    
    E_dirs = []
    if 'x' in E_input:
        E_dirs.append('x')
    if 'y' in E_input:
        E_dirs.append('y')
    if 'z' in E_input:
        E_dirs.append('z')
    
    if len(E_dirs) == 0:
        E_dirs = ['z']
    
    print("\n")
    print_header("Selection Rules")
    print("Delta_l = +/-1")
    print("Delta_ms = 0")
    print("E field components: {}".format(', '.join(E_dirs)))
    for E_dir in E_dirs:
        if E_dir == 'z':
            print("  z: Delta_ml = 0 (q=0)")
        else:
            print("  {}: Delta_ml = +/-1 (q=+/-1)".format(E_dir))
    input("Press Enter...")
    
    while True:
        print("\n")
        print_header("Test Final State")
        print("Enter final state (or 'm' for menu):")
        np_input = input("n' (or m): ").strip()
        if np_input.lower() == 'm':
            break
        
        np = int(np_input)
        lp = int(input("l': ").strip())
        mlp = int(input("ml': ").strip())
        msp_input = input("ms' (+ or -): ").strip()
        msp = 0.5 if '+' in msp_input else -0.5
        
        allowed, reasons, allowed_dirs = check_dipole_selection(n, l, ml, ms, np, lp, mlp, msp, E_dirs)
        
        print("\n")
        print_header("Result")
        print("|n'={}, l'={}, ml'={}, ms'={}>".format(np, lp, mlp, format_frac(msp)))
        print("")
        if allowed:
            print("ALLOWED")
            print("")
            for reason in reasons:
                print(reason)
            
            input("Press Enter...")
            print("\n")
            print_header("Transition Amplitude")
            
            print("<n'={}, l'={}, ml'={}, ms'={}|".format(np, lp, mlp, format_frac(msp)))
            print(" e*r_q |n={}, l={}, ml={}, ms={}>".format(n, l, ml, format_frac(ms)))
            print("")
            
            for E_dir, q in allowed_dirs:
                cg_coeff = clebsch_gordan(l, ml, 1, q, lp, mlp)
                print("For E||{} (q={}):".format(E_dir, q))
                print("  <l={},ml={}; 1,q={} | l'={},ml'={}>".format(l, ml, q, lp, mlp))
                print("  * <n'={},l'={}|| e*r ||n={},l={}>".format(np, lp, n, l))
                print("  CG coeff = {}".format(format_frac(cg_coeff)))
                print("")
        else:
            print("NOT ALLOWED")
            print("")
            for reason in reasons:
                print(reason)
            print("")
        
        input("Press Enter...")
    
    print("\n")
    restart = input("New state? (y/n): ").strip().lower()
    if restart != 'y':
        break

print("Goodbye!")
