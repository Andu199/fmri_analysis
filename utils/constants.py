# BASIC_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter", "global_signal"]
# DERIVATIVES_OF_BASIC = [con + "_derivative1" for con in BASIC_CONFOUNDS]
# POWER_2_BASIC = [con + "_power2" for con in (BASIC_CONFOUNDS + DERIVATIVES_OF_BASIC)]
from enum import Enum

BASIC_CONFOUNDS = ["X", "Y", "Z", "RotX", "RotY", "RotZ", "WhiteMatter", "GlobalSignal"]
DERIVATIVES_OF_BASIC = [con + "_derivative1" for con in BASIC_CONFOUNDS]
POWER_2_BASIC = [con + "_power2" for con in (BASIC_CONFOUNDS + DERIVATIVES_OF_BASIC)]

CONFOUNDS_DICT = {
    "basic_confounds": BASIC_CONFOUNDS,
    "derivatives_of_basic": DERIVATIVES_OF_BASIC,
    "power_2_basic": POWER_2_BASIC
}


class Diagnostic(Enum):
    HEALTHY = 0
    SCHZ = 1
    BIPOLAR = 2
    ADHD = 3
