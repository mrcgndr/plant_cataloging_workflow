import numpy as np


def osaviImage(
        R: np.ndarray,
        NIR: np.ndarray,
        y_osavi: float
    ) -> np.ndarray:
    osavi = np.float16((NIR-R)/(NIR+R+y_osavi))
    return osavi

def ngrdiImage(
        R: np.ndarray,
        G: np.ndarray
    ) -> np.ndarray:
    ngrdi = np.float16((G-R)/(G+R+1e-12))
    return ngrdi

def gliImage(
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
    gli = np.float16((2*G-R-B)/(2*G+R+B+1e-12))
    return gli
