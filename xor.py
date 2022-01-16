



import numpy as np 

def activator(x):
    return 0 if x <= 0 else 1

def startnet(C):
    x = np.array([C[0], C[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])
    
    sum = np.dot(w_hidden, x)
    out = [activator(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = activator(sum)
    return y

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

print( startnet(C1[0]), startnet(C1[1]) )
print( startnet(C2[0]), startnet(C2[1]) )

