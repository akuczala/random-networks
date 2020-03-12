import numpy as np

#code for generating block-structured random matrices
def generateDiagBlock(sig,rho,blockSize,complex = False):
    blockSize = int(blockSize)
    #zero condition
    if sig == 0:
        return np.zeros([blockSize,blockSize])
    #create diagonal entries
    if complex:
        block = np.random.normal(0,sig*np.sqrt(1+rho)/np.sqrt(2),[blockSize,blockSize]) \
            + 1j*np.random.normal(0,sig*np.sqrt(1-rho)/np.sqrt(2),[blockSize,blockSize])
        covarRe = np.array([[sig*sig,rho*sig*sig],[rho*sig*sig,sig*sig]])/2
        covarIm = np.array([[sig*sig,-rho*sig*sig],[-rho*sig*sig,sig*sig]])/2
        offdiag = np.transpose(np.random.multivariate_normal([0,0],covarRe,[int(blockSize*(blockSize-1)/2)])) \
            + 1j*np.transpose(np.random.multivariate_normal([0,0],covarIm,[int(blockSize*(blockSize-1)/2)]))
    else:
        covar = np.array([[sig*sig,rho*sig*sig],[rho*sig*sig,sig*sig]])
        block = np.random.normal(0,sig,[int(blockSize),int(blockSize)])
        offdiag = np.transpose(np.random.multivariate_normal([0,0],covar,[int(blockSize*(blockSize-1)/2)]))
    uIndex = np.triu_indices(blockSize,1)
    block[uIndex] = offdiag[0]
    block = block.T
    block[uIndex] = offdiag[1]
    return block
import numpy as np
def generateOffDiagBlockPair(sigA,sigB,rhoAB,blockASize,blockBSize, complex = False):
    blockASize = int(blockASize)
    blockBSize = int(blockBSize)
    if complex:
        covarRe = np.array([[sigA*sigA,rhoAB*sigA*sigB],[rhoAB*sigA*sigB,sigB*sigB]])/2
        covarIm = np.array([[sigA*sigA,-rhoAB*sigA*sigB],[-rhoAB*sigA*sigB,sigB*sigB]])/2
        offdiag = np.transpose(np.random.multivariate_normal([0,0],covarRe,[blockASize,blockBSize])) \
            + 1j*np.transpose(np.random.multivariate_normal([0,0],covarIm,[blockASize,blockBSize]))
    else:
        covar = np.array([[sigA*sigA,rhoAB*sigA*sigB],[rhoAB*sigA*sigB,sigB*sigB]])
        offdiag = np.transpose(np.random.multivariate_normal([0,0],covar,[blockASize,blockBSize]))
    blockAB = offdiag[0]
    blockBA = np.transpose(offdiag[1])
    return [blockAB,blockBA]
#generate blocks
def generateBlockMatrix(popSizes,gmat,tau, complex = False):
    popSizes = np.array(popSizes)
    n = int(np.sum(popSizes))
    m = popSizes.size
    cumPopSizes = np.concatenate(([0],np.cumsum(popSizes))).astype(int)
    if complex:
        J = np.zeros([n,n],dtype = np.complex_)
    else:
        J = np.zeros([n,n])
    for k1 in range(m):
        for k2 in range(k1+1):
            #print k1, k2
            if k1 == k2:
                k = k1
                J[cumPopSizes[k]:cumPopSizes[k+1],cumPopSizes[k]:cumPopSizes[k+1]] = generateDiagBlock(gmat[k,k]/np.sqrt(n),tau[k,k],popSizes[k], complex = complex)
            else:
                [B,C] = generateOffDiagBlockPair(gmat[k1,k2]/np.sqrt(n),gmat[k2,k1]/np.sqrt(n),tau[k1,k2],popSizes[k2],popSizes[k1], complex = complex)
                J[cumPopSizes[k1]:cumPopSizes[k1+1],cumPopSizes[k2]:cumPopSizes[k2+1]] = B
                J[cumPopSizes[k2]:cumPopSizes[k2+1],cumPopSizes[k1]:cumPopSizes[k1+1]] = C
    return J
#generate a matrix where the variance and correlation depend on the index
#useful for approximating continuously-varying statistics
def generateMatrix_ij(g_ij,tau_ij,complex = False):
    n = g_ij.shape[0]
    if complex:
        J = np.random.normal(0,1,[n,n])/np.sqrt(2*n) + 1.j*np.random.normal(0,1,[n,n])/np.sqrt(2*n)
    else:
        J = np.random.normal(0,1,[n,n])/np.sqrt(n)
    for i in range(n):
        for j in range(i):
            g1 = g_ij[i,j]; g2 = g_ij[j,i]
            rho = tau_ij[i,j]
            if complex:
                pair = [J[i,j],J[j,i]]
                J[i,j] = g1*pair[0]
                J[j,i] = g2*(rho*np.conj(pair[0])+np.sqrt(1.-rho*rho)*pair[1])
            else:
                pair = np.array([J[i,j],J[j,i]])
                J[i,j] = g1*pair[0]
                J[j,i] =  g2*(rho*pair[0]+np.sqrt(1.-rho*rho)*pair[1])
    return J