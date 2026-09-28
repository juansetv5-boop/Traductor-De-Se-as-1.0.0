import os
import re

def patch_validator():
    path = r"src\core\hand_pose_validator.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update DYNAMIC_ALPHABET_LETTERS
    content = content.replace(
        'DYNAMIC_ALPHABET_LETTERS: List[str] = ["j", "z"]',
        'DYNAMIC_ALPHABET_LETTERS: List[str] = ["j", "z", "ñ"]'
    )

    # 2. Add geometric tags to CANONICAL_ALPHABET_MATRIX
    tags = {
        '"e": {': '"e": {\n        "geometric": "thumb_under_fingers",',
        '"h": {': '"h": {\n        "geometric": "horizontal_index_middle",',
        '"m": {': '"m": {\n        "geometric": "thumb_under_fingers",',
        '"n": {': '"n": {\n        "geometric": "thumb_under_fingers",',
        '"o": {': '"o": {\n        "geometric": "o_shape",',
        '"p": {': '"p": {\n        "geometric": "pointing_down",',
        '"q": {': '"q": {\n        "geometric": "pointing_down",',
        '"r": {': '"r": {\n        "geometric": "crossed_fingers",',
        '"s": {': '"s": {\n        "geometric": "thumb_over_fingers",',
        '"t": {': '"t": {\n        "geometric": "thumb_between_index_middle",',
        '"x": {': '"x": {\n        "geometric": "index_hook",'
    }
    for k, v in tags.items():
        # Match only the entry start inside the matrix to avoid false positives
        content = re.sub(r'(' + re.escape(k) + r'\s+"fingers": \{[^\}]+\},\s+"desc": "[^"]+",)', v.replace(k, k[0:6]) + r'\n\1'[6:], content) # manual inject

    # Wait, simple string replace is safer if we just replace the desc line
    # Let's do it cleanly by replacing desc lines:
    desc_replacements = {
        '"desc": "Todos los dedos flexionados descansando sobre el pulgar",': '"desc": "Todos los dedos flexionados descansando sobre el pulgar",\n        "geometric": "thumb_under_fingers",',
        '"desc": "Índice y medio extendidos juntos horizontalmente",': '"desc": "Índice y medio extendidos juntos horizontalmente",\n        "geometric": "horizontal_index_middle",',
        '"desc": "Tres dedos flexionados cubriendo el pulgar",': '"desc": "Tres dedos flexionados cubriendo el pulgar",\n        "geometric": "thumb_under_fingers",',
        '"desc": "Dos dedos flexionados cubriendo el pulgar",': '"desc": "Dos dedos flexionados cubriendo el pulgar",\n        "geometric": "thumb_under_fingers",',
        '"desc": "Dedos curvados tocando la punta del pulgar formando un círculo \'O\'",': '"desc": "Dedos curvados tocando la punta del pulgar formando un círculo \'O\'",\n        "geometric": "o_shape",',
        '"desc": "Configuración \'K\' orientada hacia abajo",': '"desc": "Configuración \'K\' orientada hacia abajo",\n        "geometric": "pointing_down",',
        '"desc": "Configuración \'G\' apuntando hacia abajo",': '"desc": "Configuración \'G\' apuntando hacia abajo",\n        "geometric": "pointing_down",',
        '"desc": "Índice y medio extendidos y cruzados",': '"desc": "Índice y medio extendidos y cruzados",\n        "geometric": "crossed_fingers",',
        '"desc": "Puño con pulgar cruzado horizontalmente sobre los dedos",': '"desc": "Puño con pulgar cruzado horizontalmente sobre los dedos",\n        "geometric": "thumb_over_fingers",',
        '"desc": "Puño con pulgar insertado entre índice y medio",': '"desc": "Puño con pulgar insertado entre índice y medio",\n        "geometric": "thumb_between_index_middle",',
        '"desc": "Índice flexionado en gancho, demás dedos cerrados",': '"desc": "Índice flexionado en gancho, demás dedos cerrados",\n        "geometric": "index_hook",',
    }
    for old, new in desc_replacements.items():
        content = content.replace(old, new)

    # 3. Add Ñ
    n_entry = '''"n": {
        "fingers": {"thumb": "CURLED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Dos dedos flexionados cubriendo el pulgar",
        "geometric": "thumb_under_fingers",
    },'''
    n_ñ_entry = n_entry + '''
    "ñ": {
        "fingers": {"thumb": "CURLED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Dos dedos flexionados cubriendo el pulgar con oscilación lateral",
        "geometric": "thumb_under_fingers",
        "is_dynamic": True,
    },'''
    content = content.replace(n_entry, n_ñ_entry)

    # 4. Replace geometric validations
    old_val = """    # Validaciones geométricas de precisión para diferenciar letras análogas
    geometric = spec.get("geometric")
    if geometric == "v_spread":
        # 'V': Índice y Medio extendidos Y separados
        d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
        if d_im < 0.040:
            return False, "index", "Corrige: Separa el Índice y Medio en forma de 'V'", finger_states

    elif geometric == "u_together":
        # 'U': Índice y Medio extendidos Y juntos
        d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
        if d_im > 0.080:
            return False, "index", "Corrige: Junta los dedos Índice y Medio para 'U'", finger_states

    return True, None, None, finger_states"""

    new_val = """    # Validaciones geométricas de precisión para diferenciar letras análogas
    geometric = spec.get("geometric")
    if geometric:
        if geometric == "v_spread":
            d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
            if d_im < 0.040:
                return False, "index", "Corrige: Separa el Índice y Medio en forma de 'V'", finger_states

        elif geometric == "u_together":
            d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
            if d_im > 0.080:
                return False, "index", "Corrige: Junta los dedos Índice y Medio para 'U'", finger_states

        elif geometric == "crossed_fingers":
            d_cross = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_PIP, :2]))
            if d_cross > 0.06:
                return False, "index", "Corrige: Cruza el dedo Índice sobre el Medio para la 'R'", finger_states

        elif geometric == "pointing_down":
            if pts[LM_INDEX_TIP, 1] < pts[LM_WRIST, 1]:
                return False, "index", "Corrige: Apunta los dedos hacia abajo", finger_states

        elif geometric == "horizontal_index_middle":
            dx = abs(pts[LM_INDEX_TIP, 0] - pts[LM_WRIST, 0])
            dy = abs(pts[LM_INDEX_TIP, 1] - pts[LM_WRIST, 1])
            if dy > dx:
                return False, "index", "Corrige: Apunta los dedos horizontalmente para la 'H'", finger_states

        elif geometric == "o_shape":
            d_o = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_THUMB_TIP, :2]))
            if d_o > 0.08:
                return False, "thumb", "Corrige: Junta las puntas formando un círculo 'O'", finger_states

        elif geometric == "thumb_over_fingers":
            d_s = float(np.linalg.norm(pts[LM_THUMB_TIP, :2] - pts[LM_MIDDLE_PIP, :2]))
            if d_s > 0.12:
                return False, "thumb", "Corrige: Cierra el pulgar sobre los dedos (Puño 'S')", finger_states

        elif geometric == "thumb_between_index_middle":
            if pts[LM_THUMB_TIP, 1] > pts[LM_INDEX_MCP, 1]:
                return False, "thumb", "Corrige: Inserta el pulgar entre Índice y Medio para 'T'", finger_states

        elif geometric == "thumb_under_fingers":
            if pts[LM_THUMB_TIP, 1] < pts[LM_INDEX_PIP, 1]:
                return False, "thumb", "Corrige: Oculta el pulgar debajo de los dedos", finger_states

        elif geometric == "index_hook":
            d_tip = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_INDEX_MCP, :2]))
            d_pip = float(np.linalg.norm(pts[LM_INDEX_PIP, :2] - pts[LM_INDEX_MCP, :2]))
            if d_tip > d_pip:
                return False, "index", "Corrige: Flexiona el Índice en forma de gancho 'X'", finger_states

    return True, None, None, finger_states"""

    content = content.replace(old_val, new_val)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("hand_pose_validator.py patched successfully.")


def patch_ghost_trainer():
    path = r"src\ui\ghost_trainer.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update alphabet signs init
    content = content.replace(
        'self.alphabet_signs = list("abcdefghijklmnopqrstuvwxyz")',
        'self.alphabet_signs = list("abcdefghijklmnñopqrstuvwxyz")'
    )

    # 2. Update waypoints in ABECEDARIO mode
    old_abecedario = """        if self.training_mode == "ABECEDARIO":
            # ── MODO ALFABETO: Único Ghost Anchor fijo a la altura del hombro dominante ──
            anchor_x = mid_x + 1.20 * fs
            anchor_y = mid_y - 0.20 * fs
            coords_specs = [(anchor_x, anchor_y, "Sostener Señal", False, -1, "", False, False)]"""
    
    new_abecedario = """        if self.training_mode == "ABECEDARIO":
            letra = gesture_clean
            if letra == "j":
                coords_specs = [
                    (nx + 0.8 * fs, ny, "1. Sien (Inicio J)", False, -1, "", False, False),
                    (nx + 0.8 * fs, ny + 0.8 * fs, "2. Abajo", False, -1, "", False, False),
                    (nx, ny + 0.8 * fs, "3. Gancho (Fin J)", False, -1, "", False, False),
                ]
            elif letra == "z":
                coords_specs = [
                    (mid_x - 0.5 * fs, ny, "1. Inicio Z", False, -1, "", False, False),
                    (mid_x + 0.5 * fs, ny, "2. Trazo horiz.", False, -1, "", False, False),
                    (mid_x - 0.5 * fs, ny + 0.8 * fs, "3. Diagonal", False, -1, "", False, False),
                    (mid_x + 0.5 * fs, ny + 0.8 * fs, "4. Fin Z", False, -1, "", False, False),
                ]
            elif letra == "ñ":
                coords_specs = [
                    (mid_x, ny + 0.5 * fs, "1. Centro (Ñ)", False, -1, "", False, False),
                    (mid_x + 0.6 * fs, ny + 0.5 * fs, "2. Derecha", False, -1, "", False, False),
                    (mid_x, ny + 0.5 * fs, "3. Centro", False, -1, "", False, False),
                ]
            else:
                # ── MODO ALFABETO: Único Ghost Anchor fijo a la altura del hombro dominante ──
                anchor_x = mid_x + 1.20 * fs
                anchor_y = mid_y - 0.20 * fs
                coords_specs = [(anchor_x, anchor_y, "Sostener Señal", False, -1, "", False, False)]"""
    
    content = content.replace(old_abecedario, new_abecedario)

    # 3. Update static hold timer logic
    old_hover = """        if is_hovering:
            req_time = self.initial_hold_time if target_btn.id == 0 else self.dwell_activation_time
            fill_rate = 1.0 / max(1e-3, req_time)"""
            
    new_hover = """        if is_hovering:
            req_time = self.initial_hold_time if target_btn.id == 0 else self.dwell_activation_time
            # Timer extendido (0.8s) para letras estáticas del abecedario
            if self.training_mode == "ABECEDARIO" and target_btn.id == 0 and len(self.buttons) == 1:
                req_time = 0.80

            fill_rate = 1.0 / max(1e-3, req_time)"""
            
    content = content.replace(old_hover, new_hover)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("ghost_trainer.py patched successfully.")


def patch_tests():
    path = r"tests\test_hand_pose_validator.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix create_fist_hand
    old_fist = """def create_fist_hand() -> np.ndarray:
    \"\"\"Puño cerrado con los 4 dedos completamente flexionados hacia la palma.\"\"\"
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.5, 0.8, 0.0]  # muñeca
    for i, mcp in enumerate([2, 5, 9, 13, 17]):
        hand[mcp] = [0.4 + i * 0.05, 0.6, 0.0]
    for i, pip in enumerate([3, 6, 10, 14, 18]):
        hand[pip] = [0.4 + i * 0.05, 0.52, 0.05]
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        hand[tip] = [0.4 + i * 0.05, 0.58, 0.02]
    return hand"""

    new_fist = """def create_fist_hand() -> np.ndarray:
    \"\"\"Puño cerrado con los 4 dedos completamente flexionados hacia la palma.\"\"\"
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.5, 0.8, 0.0]  # muñeca
    # Pulgar (1,2,3,4) flexionado/descansando
    hand[1] = [0.45, 0.75, 0.0]
    hand[2] = [0.42, 0.70, 0.0]
    hand[3] = [0.40, 0.65, 0.0]
    hand[4] = [0.38, 0.60, 0.0]
    for i, mcp in enumerate([5, 9, 13, 17]):
        hand[mcp] = [0.4 + i * 0.05, 0.6, 0.0]
    for i, pip in enumerate([6, 10, 14, 18]):
        hand[pip] = [0.4 + i * 0.05, 0.52, 0.05]
    for i, dip in enumerate([7, 11, 15, 19]):
        hand[dip] = [0.4 + i * 0.05, 0.55, 0.03]
    for i, tip in enumerate([8, 12, 16, 20]):
        hand[tip] = [0.4 + i * 0.05, 0.58, 0.02]
    return hand"""

    content = content.replace(old_fist, new_fist)

    # Append tests
    test_append = """

def test_alphabet_validation():
    from src.core.hand_pose_validator import validate_alphabet_letter
    
    # 1. Test geometric 'v' vs 'u'
    hand = create_fist_hand()
    # Extender índice y medio juntos ('u')
    hand[6:9] = [[0.45, 0.45, 0.0], [0.45, 0.35, 0.0], [0.45, 0.25, 0.0]]
    hand[10:13] = [[0.46, 0.45, 0.0], [0.46, 0.35, 0.0], [0.46, 0.25, 0.0]]
    
    mock_lms = MockHandLandmarks(hand)
    
    # U should pass
    ok, err_f, msg, _ = validate_alphabet_letter("u", mock_lms)
    assert ok is True
    
    # V should fail due to geometry (too close)
    ok, err_f, msg, _ = validate_alphabet_letter("v", mock_lms)
    assert ok is False
    assert err_f == "index"
    
    # Separar para 'v'
    hand[6:9] = [[0.40, 0.45, 0.0], [0.38, 0.35, 0.0], [0.35, 0.25, 0.0]]
    hand[10:13] = [[0.50, 0.45, 0.0], [0.52, 0.35, 0.0], [0.55, 0.25, 0.0]]
    mock_lms = MockHandLandmarks(hand)
    
    # V should pass now
    ok, err_f, msg, _ = validate_alphabet_letter("v", mock_lms)
    assert ok is True
    # U should fail
    ok, err_f, msg, _ = validate_alphabet_letter("u", mock_lms)
    assert ok is False

"""
    # Just append before the __main__ block
    content = content.replace('if __name__ == "__main__":', test_append + '\nif __name__ == "__main__":')
    content = content.replace('test_resolve_active_hand_cursor_numpy_boolean_safe()', 'test_resolve_active_hand_cursor_numpy_boolean_safe()\n    test_alphabet_validation()')

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("test_hand_pose_validator.py patched successfully.")

if __name__ == "__main__":
    patch_validator()
    patch_ghost_trainer()
    patch_tests()
