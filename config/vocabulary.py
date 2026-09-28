"""
config/vocabulary.py — Catálogo oficial de vocabulario para el sistema Alpha LSC.
Define el catálogo de verbos dinámicos y vocabulario del abecedario.
"""
from typing import Dict, Final, List

# Catálogo oficial de verbos dinámicos en LSC
# Se diferencian direccionalmente las dos clases de ayuda:
# - "ayudar": Acción de dar ayuda (Del pecho hacia afuera ->).
# - "ayudame": Acción de pedir ayuda (De afuera hacia el pecho <-).
DYNAMIC_VERBS: Final[List[str]] = [
    "hola",
    "gracias",
    "por favor",
    "ayudar",
    "ayudame",
    "buenos dias",
    "caminar",
    "jugar",
]

# Direcciones y flechas indicativas para el selector del HUD
VERB_DIRECTIONS: Final[Dict[str, str]] = {
    "ayudar": "-> afuera",
    "ayudame": "<- adentro",
}

# Descripciones contextuales para ayuda y retroalimentación pedagógica
VERB_DESCRIPTIONS: Final[Dict[str, str]] = {
    "hola": "Saludo con mano oscilante o abierta",
    "gracias": "Toque de mentón/pecho hacia el frente",
    "por favor": "Movimiento circular sobre el pecho",
    "ayudar": "Acción de dar ayuda (Del pecho hacia afuera ->)",
    "ayudame": "Acción de pedir ayuda (De afuera hacia el pecho <-)",
    "buenos dias": "Compuesto bimanual (pecho hacia apertura)",
    "caminar": "Piernas caminantes sobre la palma base",
    "jugar": "Manos en configuración Y oscilantes",
}
