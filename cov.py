import numpy as np

def cov(evector1,evector2,evalue):
    if type(evalue) == np.ndarray:
        evalue = np.diag(evalue)
    else:
        evalue = np.array(evalue)

        evalue = np.diag(evalue)

    S = np.vstack([evector1,evector2])

    return np.dot(np.dot(S.T,evalue),S)


