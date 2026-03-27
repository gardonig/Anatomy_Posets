from dataclasses import dataclass

# Constants for the axes used throughout the application
AXIS_VERTICAL = "vertical"           # top–bottom (superior–inferior)
AXIS_MEDIOLATERAL = "mediolateral"   # right–left (lateral, patient's view)
AXIS_ANTERIOR_POSTERIOR = "anteroposterior"  # front–back (anteroposterior)

@dataclass
class Structure:
    """A node in the poset: an anatomical structure (organ, bone, muscle, etc.)."""
    name: str
    # Convention A (all in standard anatomical position, scaled to [0, 100]):
    # - com_vertical: 0 = toes/feet (inferior), 100 = head/vertex (superior)
    # - com_lateral: 0 = right side, 100 = left side (patient's view)
    # - com_anteroposterior: 0 = back (dorsal), 100 = front (ventral)
    com_vertical: float
    com_lateral: float
    com_anteroposterior: float