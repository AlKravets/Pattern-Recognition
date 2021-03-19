import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
# np.random.seed(11)
np.random.seed(29)
# np.random.seed(143)



l = [
    [-1,0],
    [0,-1],
    [1,0],
    [1,1]
]

l = np.array(l)

l1= [
    [1,0],
    [0,0],
    [1,1],
]

l1 = np.array(l1)


def angle (con_pol, m,n):
    mod = con_pol.shape[0]
    # print(f'first line start {con_pol[(m)%mod]}, end {con_pol[(m+1)%mod]}')
    # print(f'second line start {con_pol[(n)%mod]}, end {con_pol[(n+1)%mod]}')
    k1 = np.arctan2((con_pol[(m+1) % mod][1] - con_pol[m% mod][1]),(con_pol[(m+1) % mod][0] - con_pol[m% mod][0]))
    
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    print(f'before rotate k1 {k1}, k2 {k2}')

    k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    print(f'after rotate k1 {k1}, k2 {k2}')
    
    if k2< k1:
        res = np.pi  - (k1 - k2)
    else:
        res = np.abs(k1 - k2)

    print(f'k1 {k1}, k2 {k2}, res {res}')
    return np.abs(res)


if __name__ == '__main__':

    angle(l, 0,2)