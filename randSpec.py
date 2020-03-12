import numpy as np

#one iteration of the self-consistent equation
def fixedPointCov(a,c,d,zbar,ep,gsq,T,fracs):
    aNum = ep+np.dot(a*fracs,gsq)
    dNum = ep+np.dot(gsq,d*fracs)
    cNum = -np.dot(T,np.conj(c)*fracs)+zbar
    bNum = np.conj(cNum)
    q = (aNum)*(dNum) + (bNum)*(cNum)
    return np.array([aNum,cNum,dNum])/q

#iterate self consistent equations to compute green's function at a point z
def trGreenIterate(gsq,T,fracs,z,max_iterations,ep=0,safe=True,returnC = True,printWarning=False,returnIterations=False,threshold = 1E-6):
    m = fracs.shape[0]
    init = np.ones([m])/np.sqrt(m)
    zbar = np.conj(z)
    a = init.copy(); c = init.copy(); d = init.copy()
    for it in range(max_iterations):
        lastc = c.copy()
        [a,c,d] = fixedPointCov(a,c,d,zbar,ep,gsq,T,fracs)
        #stop iterating when threshold is reached
        if np.vdot(c-lastc,c-lastc) < threshold:
            break
    if it == max_iterations - 1 and printWarning:
        print('Reached ', it, ' iterations at ', z)
    if returnIterations: #return # of iterations
        return it
    if returnC: #return trace of c
        trace = np.dot(c,fracs)
    else: #return trace/i of A + D
        trace = np.imag(np.dot(a,fracs) + np.dot(d,fracs))
    return trace
    
#compute the eigenvalue density given a grid of points in the complex plane
def calc_density_grid(gsq,T,fracs,zGrid,max_iterations,threshold=1E-6):
    greenGrid = np.vectorize(lambda z : trGreenIterate(gsq,T,fracs,z,max_iterations,returnC=True,threshold=threshold))(zGrid)
    dcy = np.diff(greenGrid)
    dcx = np.diff(greenGrid.T).T
    #average elements to make array square
    dcy = np.array(list(map(lambda row: np.convolve(row,np.ones([2])/2,'same')[1:],dcy.T))).T
    dcx = np.array(list(map(lambda row: np.convolve(row,np.ones([2])/2,'same')[1:],dcx)))
    dx = np.abs(np.real(zGrid[0,0]-zGrid[1,0]))
    dy = np.abs(np.imag(zGrid[0,0]-zGrid[0,1]))
    dzbar = 0.5*(dcx/dx+1j*dcy/dy)
    return np.real(dzbar)/(np.pi)