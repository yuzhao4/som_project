import numpy as np

def find_angle(v2,v1 = np.array([1,1,1]),original = np.array([0,0,0])):
    '''
    : v2, v1           The second vector and the first vector
    : original         The zero point to normalize two vectors
    '''
    v2 = np.array(v2)
    v1 = np.array(v1)
        
    v2 = v2 - original
    v1 = v1 - original

    cosine_angle = np.dot(v2,v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print('dot product is ',(np.dot(v2,v1)))
    print('v1 norm is',np.linalg.norm(v1))
    print('v2 norm is',np.linalg.norm(v2))
    print('cos angle is ',cosine_angle)
    if np.absolute(cosine_angle  - 1) < 0.0001:
        return 0
    else:
        return np.arccos(cosine_angle) * 180 / np.pi

v2 = np.array([0,1,0])

v1 = np.array([1,0,0])

print(find_angle(v2,v1))
