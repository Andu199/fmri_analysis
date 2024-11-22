# BASIC_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter", "global_signal"]
# DERIVATIVES_OF_BASIC = [con + "_derivative1" for con in BASIC_CONFOUNDS]
# POWER_2_BASIC = [con + "_power2" for con in (BASIC_CONFOUNDS + DERIVATIVES_OF_BASIC)]
from enum import Enum

BASIC_CONFOUNDS = ["X", "Y", "Z", "RotX", "RotY", "RotZ", "WhiteMatter", "GlobalSignal"]
# BASIC_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "white_matter", "global_signal"]

DERIVATIVES_OF_BASIC = [con + "_derivative1" for con in BASIC_CONFOUNDS]
POWER_2_BASIC = [con + "_power2" for con in (BASIC_CONFOUNDS + DERIVATIVES_OF_BASIC)]
COMPCOR = ["tCompCor00", "tCompCor01", "tCompCor02", "tCompCor03", "tCompCor04", "tCompCor05",
           "aCompCor00", "aCompCor01", "aCompCor02", "aCompCor03", "aCompCor04", "aCompCor05"]

CONFOUNDS_DICT = {
    "basic_confounds": BASIC_CONFOUNDS,
    "derivatives_of_basic": DERIVATIVES_OF_BASIC,
    "power_2_basic": POWER_2_BASIC,
    "compcor": COMPCOR,
}


class Diagnostic(Enum):
    HEALTHY = 0
    SCHZ = 1
    BIPOLAR = 2
    ADHD = 3
    HPAIN = 4
    PAIN = 5

STRING_TO_DIAGNOSTIC = {
    "h": Diagnostic.HEALTHY,
    "healthy": Diagnostic.HEALTHY,
    "schz": Diagnostic.SCHZ,
    "bipolar": Diagnostic.BIPOLAR,
    "adhd": Diagnostic.ADHD,
    "hpain": Diagnostic.HPAIN,
    "pain": Diagnostic.PAIN
}