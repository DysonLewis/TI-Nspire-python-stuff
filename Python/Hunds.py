# TI-Nspire compatible Atomic Term Symbol Calculator

def get_element_data():
    elements = {
        'h': (1, 'hydrogen', '1s1'), 'he': (2, 'helium', '1s2'),
        'li': (3, 'lithium', '[He]2s1'), 'be': (4, 'beryllium', '[He]2s2'),
        'b': (5, 'boron', '[He]2s2 2p1'), 'c': (6, 'carbon', '[He]2s2 2p2'),
        'n': (7, 'nitrogen', '[He]2s2 2p3'), 'o': (8, 'oxygen', '[He]2s2 2p4'),
        'f': (9, 'fluorine', '[He]2s2 2p5'), 'ne': (10, 'neon', '[He]2s2 2p6'),
        'na': (11, 'sodium', '[Ne]3s1'), 'mg': (12, 'magnesium', '[Ne]3s2'),
        'al': (13, 'aluminum', '[Ne]3s2 3p1'), 'si': (14, 'silicon', '[Ne]3s2 3p2'),
        'p': (15, 'phosphorus', '[Ne]3s2 3p3'), 's': (16, 'sulfur', '[Ne]3s2 3p4'),
        'cl': (17, 'chlorine', '[Ne]3s2 3p5'), 'ar': (18, 'argon', '[Ne]3s2 3p6'),
        'k': (19, 'potassium', '[Ar]4s1'), 'ca': (20, 'calcium', '[Ar]4s2'),
        'sc': (21, 'scandium', '[Ar]3d1 4s2'), 'ti': (22, 'titanium', '[Ar]3d2 4s2'),
        'v': (23, 'vanadium', '[Ar]3d3 4s2'), 'cr': (24, 'chromium', '[Ar]3d5 4s1'),
        'mn': (25, 'manganese', '[Ar]3d5 4s2'), 'fe': (26, 'iron', '[Ar]3d6 4s2'),
        'co': (27, 'cobalt', '[Ar]3d7 4s2'), 'ni': (28, 'nickel', '[Ar]3d8 4s2'),
        'cu': (29, 'copper', '[Ar]3d10 4s1'), 'zn': (30, 'zinc', '[Ar]3d10 4s2'),
        'ga': (31, 'gallium', '[Ar]3d10 4s2 4p1'), 'ge': (32, 'germanium', '[Ar]3d10 4s2 4p2'),
        'as': (33, 'arsenic', '[Ar]3d10 4s2 4p3'), 'se': (34, 'selenium', '[Ar]3d10 4s2 4p4'),
        'br': (35, 'bromine', '[Ar]3d10 4s2 4p5'), 'kr': (36, 'krypton', '[Ar]3d10 4s2 4p6'),
        'rb': (37, 'rubidium', '[Kr]5s1'), 'sr': (38, 'strontium', '[Kr]5s2'),
        'y': (39, 'yttrium', '[Kr]4d1 5s2'), 'zr': (40, 'zirconium', '[Kr]4d2 5s2'),
        'nb': (41, 'niobium', '[Kr]4d4 5s1'), 'mo': (42, 'molybdenum', '[Kr]4d5 5s1'),
        'tc': (43, 'technetium', '[Kr]4d5 5s2'), 'ru': (44, 'ruthenium', '[Kr]4d7 5s1'),
        'rh': (45, 'rhodium', '[Kr]4d8 5s1'), 'pd': (46, 'palladium', '[Kr]4d10'),
        'ag': (47, 'silver', '[Kr]4d10 5s1'), 'cd': (48, 'cadmium', '[Kr]4d10 5s2'),
        'in': (49, 'indium', '[Kr]4d10 5s2 5p1'), 'sn': (50, 'tin', '[Kr]4d10 5s2 5p2'),
        'sb': (51, 'antimony', '[Kr]4d10 5s2 5p3'), 'te': (52, 'tellurium', '[Kr]4d10 5s2 5p4'),
        'i': (53, 'iodine', '[Kr]4d10 5s2 5p5'), 'xe': (54, 'xenon', '[Kr]4d10 5s2 5p6'),
        'cs': (55, 'cesium', '[Xe]6s1'), 'ba': (56, 'barium', '[Xe]6s2'),
        'la': (57, 'lanthanum', '[Xe]5d1 6s2'), 'ce': (58, 'cerium', '[Xe]4f1 5d1 6s2'),
        'pr': (59, 'praseodymium', '[Xe]4f3 6s2'), 'nd': (60, 'neodymium', '[Xe]4f4 6s2'),
        'pm': (61, 'promethium', '[Xe]4f5 6s2'), 'sm': (62, 'samarium', '[Xe]4f6 6s2'),
        'eu': (63, 'europium', '[Xe]4f7 6s2'), 'gd': (64, 'gadolinium', '[Xe]4f7 5d1 6s2'),
        'tb': (65, 'terbium', '[Xe]4f9 6s2'), 'dy': (66, 'dysprosium', '[Xe]4f10 6s2'),
        'ho': (67, 'holmium', '[Xe]4f11 6s2'), 'er': (68, 'erbium', '[Xe]4f12 6s2'),
        'tm': (69, 'thulium', '[Xe]4f13 6s2'), 'yb': (70, 'ytterbium', '[Xe]4f14 6s2'),
        'lu': (71, 'lutetium', '[Xe]4f14 5d1 6s2'), 'hf': (72, 'hafnium', '[Xe]4f14 5d2 6s2'),
        'ta': (73, 'tantalum', '[Xe]4f14 5d3 6s2'), 'w': (74, 'tungsten', '[Xe]4f14 5d4 6s2'),
        're': (75, 'rhenium', '[Xe]4f14 5d5 6s2'), 'os': (76, 'osmium', '[Xe]4f14 5d6 6s2'),
        'ir': (77, 'iridium', '[Xe]4f14 5d7 6s2'), 'pt': (78, 'platinum', '[Xe]4f14 5d9 6s1'),
        'au': (79, 'gold', '[Xe]4f14 5d10 6s1'), 'hg': (80, 'mercury', '[Xe]4f14 5d10 6s2'),
        'tl': (81, 'thallium', '[Xe]4f14 5d10 6s2 6p1'), 'pb': (82, 'lead', '[Xe]4f14 5d10 6s2 6p2'),
        'bi': (83, 'bismuth', '[Xe]4f14 5d10 6s2 6p3'), 'po': (84, 'polonium', '[Xe]4f14 5d10 6s2 6p4'),
        'at': (85, 'astatine', '[Xe]4f14 5d10 6s2 6p5'), 'rn': (86, 'radon', '[Xe]4f14 5d10 6s2 6p6'),
        'fr': (87, 'francium', '[Rn]7s1'), 'ra': (88, 'radium', '[Rn]7s2'),
        'ac': (89, 'actinium', '[Rn]6d1 7s2'), 'th': (90, 'thorium', '[Rn]6d2 7s2'),
        'pa': (91, 'protactinium', '[Rn]5f2 6d1 7s2'), 'u': (92, 'uranium', '[Rn]5f3 6d1 7s2'),
        'np': (93, 'neptunium', '[Rn]5f4 6d1 7s2'), 'pu': (94, 'plutonium', '[Rn]5f6 7s2'),
        'am': (95, 'americium', '[Rn]5f7 7s2'), 'cm': (96, 'curium', '[Rn]5f7 6d1 7s2'),
        'bk': (97, 'berkelium', '[Rn]5f9 7s2'), 'cf': (98, 'californium', '[Rn]5f10 7s2'),
        'es': (99, 'einsteinium', '[Rn]5f11 7s2'), 'fm': (100, 'fermium', '[Rn]5f12 7s2'),
        'md': (101, 'mendelevium', '[Rn]5f13 7s2'), 'no': (102, 'nobelium', '[Rn]5f14 7s2'),
        'lr': (103, 'lawrencium', '[Rn]5f14 7s2 7p1'), 'rf': (104, 'rutherfordium', '[Rn]5f14 6d2 7s2'),
        'db': (105, 'dubnium', '[Rn]5f14 6d3 7s2'), 'sg': (106, 'seaborgium', '[Rn]5f14 6d4 7s2'),
        'bh': (107, 'bohrium', '[Rn]5f14 6d5 7s2'), 'hs': (108, 'hassium', '[Rn]5f14 6d6 7s2'),
        'mt': (109, 'meitnerium', '[Rn]5f14 6d7 7s2'), 'ds': (110, 'darmstadtium', '[Rn]5f14 6d9 7s1'),
        'rg': (111, 'roentgenium', '[Rn]5f14 6d10 7s1'), 'cn': (112, 'copernicium', '[Rn]5f14 6d10 7s2'),
        'nh': (113, 'nihonium', '[Rn]5f14 6d10 7s2 7p1'), 'fl': (114, 'flerovium', '[Rn]5f14 6d10 7s2 7p2'),
        'mc': (115, 'moscovium', '[Rn]5f14 6d10 7s2 7p3'), 'lv': (116, 'livermorium', '[Rn]5f14 6d10 7s2 7p4'),
        'ts': (117, 'tennessine', '[Rn]5f14 6d10 7s2 7p5'), 'og': (118, 'oganesson', '[Rn]5f14 6d10 7s2 7p6')
    }
    return elements

def parse_orbital(orb_str):
    n = int(orb_str[0])
    l_char = orb_str[1]
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    l = l_map[l_char]
    return n, l

def get_ml_values(l):
    return list(range(-l, l + 1))

def distribute_electrons_hund(n_electrons, l):
    ml_values = get_ml_values(l)
    max_electrons = 2 * (2 * l + 1)
    
    if n_electrons > max_electrons:
        n_electrons = max_electrons
    
    config = {}
    for ml in ml_values:
        config[ml] = []
    
    for i in range(n_electrons):
        if i < len(ml_values):
            config[ml_values[i]].append(0.5)
        else:
            idx = i - len(ml_values)
            config[ml_values[idx]].append(-0.5)
    
    return config

def calculate_total_angular_momentum(config):
    total_L = 0
    total_S = 0
    
    for ml, spins in config.items():
        total_L += ml * len(spins)
        for s in spins:
            total_S += s
    
    return abs(total_L), abs(total_S)

def get_term_symbol(L, S):
    L_symbols = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N']
    L_symbol = L_symbols[L] if L < len(L_symbols) else str(L)
    multiplicity = int(2 * S + 1)
    
    J_values = []
    min_J = abs(L - S)
    max_J = L + S
    
    current_J = min_J
    while current_J <= max_J:
        J_values.append(current_J)
        current_J += 1
    
    return L_symbol, multiplicity, J_values

def analyze_two_electron_config(n1, l1, n2, l2):
    L_values = list(range(abs(l1 - l2), l1 + l2 + 1))
    S_values = [0, 1]
    
    all_terms = []
    for S in S_values:
        for L in L_values:
            L_symbol, mult, J_vals = get_term_symbol(L, S)
            for J in J_vals:
                all_terms.append((S, L, J, L_symbol, mult))
    
    return L_values, S_values, all_terms

def get_wavefunction(n1, l1, n2, l2, S, L, J, mJ):
    l_chars = ['s', 'p', 'd', 'f']
    
    lines = []
    lines.append("^{}{}_{}, mJ={}".format(int(2*S+1), 
        ['S','P','D','F','G','H'][L], J, mJ))
    
    if S == 1:
        ml1 = mJ
        ml2 = 0
        if l1 == 0:
            ml1 = 0
            ml2 = mJ
        elif l2 == 0:
            ml1 = mJ
            ml2 = 0
        else:
            if mJ <= l1:
                ml1 = mJ
                ml2 = 0
            else:
                ml1 = l1
                ml2 = mJ - l1
        
        lines.append("Psi=(1/sqrt(2))[psi_{}{}{}(r1)*psi_{}{}{}(r2)".format(
            n1, l_chars[l1], ml1, n2, l_chars[l2], ml2))
        lines.append("-psi_{}{}{}(r2)*psi_{}{}{}(r1)]*".format(
            n1, l_chars[l1], ml1, n2, l_chars[l2], ml2))
        
        if mJ == J:
            lines.append("|up,up>")
        elif mJ == -J:
            lines.append("|down,down>")
        else:
            lines.append("(1/sqrt(2))(|up,down>+|down,up>)")
        
        lines.append("Spatial: antisym, Spin: sym")
    else:
        ml1 = mJ
        ml2 = 0
        if l1 == 0:
            ml1 = 0
            ml2 = mJ
        elif l2 == 0:
            ml1 = mJ
            ml2 = 0
        else:
            if abs(mJ) <= l1:
                ml1 = mJ
                ml2 = 0
            else:
                ml1 = l1 if mJ > 0 else -l1
                ml2 = mJ - ml1
        
        lines.append("Psi=(1/sqrt(2))[psi_{}{}{}(r1)*psi_{}{}{}(r2)".format(
            n1, l_chars[l1], ml1, n2, l_chars[l2], ml2))
        lines.append("+psi_{}{}{}(r2)*psi_{}{}{}(r1)]*".format(
            n1, l_chars[l1], ml1, n2, l_chars[l2], ml2))
        lines.append("(1/sqrt(2))(|up,down>-|down,up>)")
        lines.append("Spatial: sym, Spin: antisym")
    
    return lines

def print_section_header(text):
    print("=" * 30)
    print(text)
    print("=" * 30)

def clear_screen_simulation():
    print("\n")

print("Atomic Term Symbol Calculator")
print("=" * 30)

elements = get_element_data()

while True:
    element_input = input("Element (symbol/name) or q: ").strip().lower()
    
    if element_input == 'q':
        break
    
    element_key = None
    for key, (z, name, config) in elements.items():
        if key == element_input or name == element_input:
            element_key = key
            break
    
    if not element_key:
        print("Element not found")
        continue
    
    z, name, full_config = elements[element_key]
    name_display = name[0].upper() + name[1:] if len(name) > 0 else name
    print("Element: {} (Z={})".format(name_display, z))
    
    clear_screen_simulation()
    print_section_header("Analysis Type")
    print("1) Ground state")
    print("2) Excited state")
    analysis_type = input("Choice: ").strip()
    
    if analysis_type == '1':
        clear_screen_simulation()
        print_section_header("Ground State Analysis")
        
        valence_config = full_config.split(']')[-1].strip()
        parts = valence_config.split()
        last_orbital = parts[-1]
        
        orbital_type = last_orbital[:2]
        n_electrons = int(last_orbital[2:])
        
        n, l = parse_orbital(orbital_type)
        
        print("Valence: {}^{}".format(orbital_type, n_electrons))
        
        if n_electrons == 2 and l == 0:
            print("Two e- in s orbital")
            print("Possible S: 0 or 1")
            print("Only valence e- contribute")
            print("(closed shells: L=0, S=0)")
            input("Press Enter...")
            clear_screen_simulation()
            print_section_header("Ground State")
            print("S=0 (antiparallel spins)")
            print("Hund's: max S, then max L")
            print("For s^2: only S=0 possible")
            print("L=0 (s orbital), J=0")
            print("Term symbol: ^1S_0")
            input("Press Enter...")
            clear_screen_simulation()
            print_section_header("Wave Function")
            print("Psi=(1/sqrt(2))*psi_{}s0(r1)*".format(n))
            print("psi_{}s0(r2)*(|up,down>-|down,up>)".format(n))
            print("Spatial: sym, Spin: antisym")
            print("Total: antisym")
            
        else:
            config = distribute_electrons_hund(n_electrons, l)
            L, S = calculate_total_angular_momentum(config)
            L_symbol, mult, J_vals = get_term_symbol(int(L), S)
            print("Electron config:")
            for ml in sorted(config.keys(), reverse=True):
                spins = config[ml]
                spin_str = ' '.join(['up' if s > 0 else 'dn' for s in spins])
                print("ml={:2d}: {}".format(ml, spin_str))
            print("L={}, S={}".format(int(L), S))
            print("Term: ^{}{}".format(mult, L_symbol))
            if len(J_vals) > 0:
                if n_electrons <= (2 * l + 1):
                    J_ground = J_vals[0]
                else:
                    J_ground = J_vals[-1]
                if J_ground == int(J_ground):
                    print("Ground J={} (Hund's 3rd)".format(int(J_ground)))
                    print("Symbol: ^{}{}_{}".format(mult, L_symbol, int(J_ground)))
                else:
                    print("Ground J={} (Hund's 3rd)".format(J_ground))
                    print("Symbol: ^{}{}_{{{}/{}}}".format(mult, L_symbol, int(J_ground*2), 2))
        
        input("Press Enter for menu...")
    
    elif analysis_type == '2':
        clear_screen_simulation()
        print_section_header("Excited State")
        
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        
        print("Electron 1 orbital:")
        n1 = int(input("n (e.g. 6): ").strip())
        l1_char = input("l (s/p/d/f): ").strip().lower()
        l1 = l_map[l1_char]
        
        print("\nElectron 2 orbital:")
        n2 = int(input("n (e.g. 6): ").strip())
        l2_char = input("l (s/p/d/f): ").strip().lower()
        l2 = l_map[l2_char]
        
        L_values, S_values, all_terms = analyze_two_electron_config(n1, l1, n2, l2)
        clear_screen_simulation()
        print_section_header("Possible States")
        print("Config: {}{}^1 {}{}^1".format(n1, l1_char, n2, l2_char))
        print("Possible L: {}".format(L_values))
        print("Possible S: 0 (singlet), 1 (triplet)")
        input("Press Enter...")
        clear_screen_simulation()
        print_section_header("Symmetry")
        print("S=1 (triplet):")
        print("  Spin: sym, Spatial: antisym")
        print("S=0 (singlet):")
        print("  Spin: antisym, Spatial: sym")
        input("Press Enter...")
        
        all_terms_sorted = sorted(all_terms, key=lambda x: (-x[2], -x[1], -x[0]))
        
        max_term = all_terms_sorted[0]
        S_max, L_max, J_max, L_sym_max, mult_max = max_term
        
        clear_screen_simulation()
        print_section_header("Max J Wavefunction")
        
        wf_lines = get_wavefunction(n1, l1, n2, l2, S_max, L_max, J_max, J_max)
        for line in wf_lines:
            print(line)
        
        input("\nPress Enter...")
        clear_screen_simulation()
        print_section_header("All Term Symbols")
        
        print("Largest J first:\n")
        for S, L, J, L_symbol, mult in all_terms_sorted:
            s_type = "triplet" if S == 1 else "singlet"
            print("^{}{}_{}  ({})".format(mult, L_symbol, J, s_type))
        
        input("\nPress Enter...")
        clear_screen_simulation()
        print_section_header("All Wavefunctions")
        
        for S, L, J, L_symbol, mult in all_terms_sorted:
            for mJ_val in range(J, -J-1, -1):
                wf_lines = get_wavefunction(n1, l1, n2, l2, S, L, J, mJ_val)
                for line in wf_lines:
                    print(line)
                print("")
                
                if mJ_val != -J:
                    cont = input("Next state (Enter) or menu (m): ").strip().lower()
                    if cont == 'm':
                        break
                    clear_screen_simulation()
                    print_section_header("All Wavefunctions")
            else:
                continue
            break
        
        input("\nPress Enter for menu...")

print("\nGoodbye!")
