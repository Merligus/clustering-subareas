import csv
import numpy as np
from numpy import linalg as LA
from scipy.optimize import nnls

from ctypes import *

class MDS:
    def __init__(self, type="ratio", ndim = 2, weight_option = 'normal', verbose = False,
                     init = "torgerson", ties = "primary", relax = False, 
                     modulus = 1, itmax = 1000, eps = 1e-6, spline_degree = 2, 
                     spline_intKnots = 2):
        self.type = type
        self.ndim = ndim
        self.weight_option = weight_option
        self.verbose = verbose
        self.init = init
        self.ties = ties
        self.relax = relax
        self.modulus = modulus
        self.itmax = itmax
        self.eps = eps
        self.spline_degree = spline_degree
        self.spline_intKnots = spline_intKnots
        
        SO_FILE = "./c_functions.so"
        self.c_functions = CDLL(SO_FILE)
    
    def mean_nan(self, diss):
        nan_numbers = np.add.reduce(np.isnan(diss), axis=(0, 1))
        s = np.nansum(diss)/2.
        n = (diss.shape[0]*diss.shape[1] - diss.shape[0] - nan_numbers)/2.
        return s/n

    def torgerson(self, delta, p = 2):
        def doubleCenter(x):
            n, m = x.shape
            s = np.sum(x) / (n * m)
            xr = np.sum(x, axis=1)/m
            xc = np.sum(x, axis=0)/n
            value = (x-np.add.outer(xc, xr)) + s
            return value
        z_values, z_vectors = LA.eigh(doubleCenter(delta**2)/2.)
        z_values *= -1
        v = np.maximum(z_values, 0.)
        if p == 1:
            normdiag = np.diag(np.sqrt(v[0]))
        else:
            normdiag = np.diag(np.sqrt(v[:p]))
            z_vectors[:,p-1] *= -1
        conf = np.dot(z_vectors[:,:p], normdiag)
        return conf

    def initConf(self, diss, n, p):
        if self.init == "torgerson":
            meandiss = self.mean_nan(diss)
            diss1 = np.copy(diss)
            diss1[np.isnan(diss1)] = meandiss
            x = self.torgerson(diss1, p = p)
        elif self.init == "random":
            x = np.random.rand(n*p, p)
        return x

    def transPrep(self, x, trans="ordinals", missing="none"):
        n = (x.shape[0]*x.shape[1] - x.shape[0])//2
        knotSeq = None
        base = None
        indices = np.triu_indices_from(x, k=1)
        iord = np.argsort(x[indices])
        y = x[indices][iord]
        y[np.isnan(y)] = np.inf

        indTieBlock = np.arange(1, n)[y[:n-1] != y[1:]]
        indTieBlock = np.insert(indTieBlock, 0, 0)
        ties = np.append(indTieBlock[1:], n) - indTieBlock
        nan_numbers = np.add.reduce(np.isnan(x[indices]))
        n_nonmis = n - nan_numbers
        iord_nonmis = iord[:n_nonmis]
        if n_nonmis < n:
            iord_mis = iord[n_nonmis:n]
            nties_nonmis = ties.shape[0] - 1
            if missing == "multiple":
                ties = np.concatenate((ties[:ties.shape[0]-1], np.repeat(1, n-n_nonmis)))
        else:
            nties_nonmis = ties.shape[0]
            iord_mis = None
        x_unique = x[indices][iord[np.cumsum(ties[:nties_nonmis])-1]]

        if trans in {"none","linear","interval"}:
            base = np.concatenate((np.matrix(np.repeat(1, nties_nonmis)), np.matrix(x_unique - x_unique[0])), axis=0).T
            xInit = np.zeros(n)
            values = np.repeat(x_unique, ties[:nties_nonmis])
            xInit.flat[iord_nonmis] = values
        elif trans in {"ordinalp","ordinals","ordinalt","ordinal","nominal","nominals","nominalp"}:
            i = np.arange(0, nties_nonmis)
            xInit = np.zeros(n)
            values = np.repeat(i, ties[:nties_nonmis])
            xInit.flat[iord_nonmis] = values
        else:
            raise TypeError(f"trans = {trans} not implemented")
        
        return {'x': x[indices],
                'x_unique': x_unique,
                'n': n,
                'n_nonmis': n_nonmis,
                'trans': trans,
                'spline_allKnots': self.spline_intKnots + 2,
                'spline_degree': self.spline_degree,
                'spline_knotSeq': knotSeq,
                'xInit': xInit,
                'iord': iord,
                'ties': ties,
                'nties_nonmis': nties_nonmis,
                'base': base,
                'missing': missing,
                'iord_nonmis': iord_nonmis,
                'iord_mis': iord_mis,
                'class': "optScal"}

    def normDissN(self, diss, wghts, m):
        n = (diss.shape[0]*diss.shape[1] - diss.shape[0])/2.
        N = n*m
        s = np.nansum(wghts*np.power(diss, 2))/2.
        # dissnorm = np.divide(diss, np.sqrt(s)*np.sqrt(N))
        dissnorm = diss/np.sqrt(s)*np.sqrt(N)
        return dissnorm
    
    def vmat(self, wghts):
        r = np.sum(wghts, axis=1)
        return np.diag(r) - wghts

    def myGenInv(self, x):
        n = x.shape[0]
        nn = 1/n
        return LA.inv(x+nn)-nn

    def bmat(self, diss, wgths, d, eps=1e-12):
        z = np.where(d < eps, 1, 0)
        b = (wgths*diss*(1-z))/(d+z)
        r = np.sum(b, axis=1)
        return np.diag(r)-b

    def weightedMean(self, y, sumwvec, target, target_ind, w, w_ind, iord, ties, n, nties):
        t_aux = target[target_ind]
        w_aux = w[w_ind]
        y = (c_double * nties)(*y)
        sumwvec = (c_double * nties)(*sumwvec)
        self.c_functions.weightedMean(y, 
                                      sumwvec, 
                                      t_aux.ctypes.data_as(c_void_p), 
                                      w_aux.ctypes.data_as(c_void_p), 
                                      iord.ctypes.data_as(c_void_p), 
                                      ties.ctypes.data_as(c_void_p), 
                                      c_long(n), 
                                      c_long(nties))
        return {"y": np.array(y[:]),
                "sumwvec": np.array(sumwvec[:])}
    
    def wmonreg(self, disp_o, w, n):
        disp_o = (c_double * n)(*disp_o)
        self.c_functions.wmonreg(disp_o, w.ctypes.data_as(c_void_p), c_long(n))
        disp_o = np.array(disp_o[:])
        return disp_o

    def transform(self, Target, x, w=None, normq=0):
        if type(w) == type(None):
            w = np.ones(len(x['x']))
        n = len(x['x'])
        b = None
        iord3 = x['iord']
        Result = np.zeros(n)
        if x['missing'] == "none":
            ind_act = x['iord_nonmis']
            nties_act = x['nties_nonmis']
        elif x['missing'] in {"single", "multiple"}:
            ind_act = x['iord']
            nties_act = len(x['ties'])
        
        y = np.zeros(nties_act)
        w2 = np.zeros(nties_act)

        Target_indices = np.triu_indices_from(Target, k=1)
        w_indices = np.triu_indices_from(w, k=1)
        d = self.weightedMean(y, w2, Target, Target_indices, w, w_indices, x['iord'], x['ties'], n, nties_act)
        y = d["y"]
        w2 = d["sumwvec"]
        
        if n > x['n_nonmis'] and x['missing'] in {"single", "multiple"}:
            Result[x['iord_mis']] = y[x['nties_nonmis']: len(x['ties'])]
        
        if x['trans'] == "none":
            Result[x['iord_nonmis']] = x['x'][x['iord_nonmis']]
        elif x['trans'] in {"linear","interval","mlinear","minterval","mspline","spline"}:
            ind_ties_nonmis = np.arange(0, x['nties_nonmis'])
            w3 = w2[ind_ties_nonmis]
            y3 = y[ind_ties_nonmis]

            ncoef = x['base'].shape[1]
            A = np.multiply(np.sqrt(w3), x['base'].T).T
            f = np.multiply(np.sqrt(w3), y3)
            if x['trans'] in {"spline","interval","linear"}:
                f = np.dot(A.T, f)
                A = np.dot(A.T, A)
                b = LA.solve(A, f.T)
            else:
                b, nnls_norm = nnls(A, f)
            Result[x['iord_nonmis']] = np.repeat(np.dot(x['base'], b), x['ties'][ind_ties_nonmis])
        elif x['trans'] in {"nominals","nominal"}:
            Result[x['iord_nonmis']] = np.repeat(y, x['ties'][np.arange(0, nties_act)])
        elif x['trans'] == "ordinalp":
            iord3 = np.argsort(x['x'])
            if n > x['n_nonmis']:
                iord3_nonmis = iord3[:x['n_nonmis']]
            else:
                iord3_nonmis = iord3
            
            Result[iord3_nonmis] = self.wmonreg(Target[Target_indices][iord3_nonmis], w[w_indices][iord3_nonmis], x['n_nonmis'])
        elif x['trans'] in {"ordinals","ordinalt","ordinal"}:
            ycon = self.wmonreg(y, w2, x['nties_nonmis'])
            ind_ties_nonmis = np.arange(0, x['nties_nonmis'])
            if x['trans'] in {"ordinals","ordinal"}:
                Result[x['iord_nonmis']] = np.repeat(ycon[ind_ties_nonmis], x['ties'][np.arange(0, x['nties_nonmis'])])
            else:
                Result[x['iord_nonmis']] = Target[Target_indices][x['iord_nonmis']] + \
                                            np.repeat(ycon[ind_ties_nonmis] - y[ind_ties_nonmis], x['ties'][np.arange(0, x['nties_nonmis'])])

        if normq > 0:
            Result = Result * np.sqrt(normq/np.sum(w[w_indices]*np.square(Result)))
        
        return {'res': Result,
                'b': b,
                'iord_prim': iord3}
    
    def spp(self, dhat, confdiss, wgths):
        resmat = np.multiply(wgths, np.square(dhat - confdiss))
        np.fill_diagonal(resmat, np.nan)
        spp = np.nanmean(resmat, axis=0)
        spp = spp/np.sum(spp)*100
        return {'spp': spp,
                'resmat': resmat}

    def fit(self, delta):
        if not np.allclose(delta, delta.T, equal_nan=True):
            raise TypeError('Distance matrix must be symmetric')

        diss = delta
        p = self.ndim
        n = diss.shape[0]
        if (p > (n - 1)):
            raise TypeError(f"Maximum number of dimensions is {n-1}!")

        nn = n*(n-1)/2
        if self.weight_option == 'normal':
            wgths = np.ones_like(diss)
        elif self.weight_option == '1or0':
            wgths = np.where(diss == np.max(diss), 0., 1.)
        elif self.weight_option == 'd-1':
            diss = np.where(diss == 0., np.inf, diss)
            d_min = np.min(diss)
            wgths = (1. / diss) / (1. / d_min)
            diss = np.where(diss == np.inf, 0, diss)
        elif self.weight_option == 'd-2':
            diss = np.where(diss == 0., np.inf, diss)
            d_min = np.min(diss)
            wgths = (1. / diss ** 2) / (1. / d_min ** 2)
            diss = np.where(diss == np.inf, 0, diss)
        else:
            raise TypeError(f"Weight option {self.weight_option} not defined.")
        wgths[np.isnan(diss)] = 0.

        x = self.initConf(diss, n, p)
        xstart = x
        trans = self.type
        if trans == "ratio":
            trans = "none"
        elif trans == "ordinal" and self.ties == "primary":
            trans = "ordinalp"
        elif trans == "ordinal" and self.ties == "secondary":
            trans = "ordinals"
        elif trans == "ordinal" and self.ties == "tertiary":
            trans = "ordinalt"
        elif trans == "spline":
            trans = "mspline"

        disobj = self.transPrep(diss, trans=trans)
        if (trans == "mspline"):
            raise TypeError(f"trans = {trans} not implemented")

        dhat = self.normDissN(diss, wgths, 1)
        dhat[np.isnan(dhat)] = 1

        if self.relax:
            self.relax = 2 
        else: 
            self.relax = 1 
        
        w = self.vmat(wgths)
        v = self.myGenInv(w)
        itel = 1
        combinations = np.indices((x.shape[0], x.shape[0]))
        # d = np.zeros((x.shape[0], x.shape[0]))
        # for i in range(x.shape[0]):
        #     for j in range(i+1, x.shape[0]):
        #         d[i, j] = np.sqrt(np.sum(np.square(x[i,:] - x[j,:])))
        #         d[j, i] = d[i, j]
        d = np.sqrt(np.sum(np.square(x[combinations[0], :] - x[combinations[1], :]), axis=2))
        lb = np.divide(np.nansum(wgths*d*dhat), np.nansum(wgths*np.square(d)))
        # lb = np.nansum(wgths*d*dhat)/np.nansum(wgths*np.square(d))
        x = lb*x
        d = lb*d

        sold = np.nansum(wgths*np.square(dhat-d))/(2*nn)
        dhat_uindices = np.triu_indices_from(dhat, k=1)
        dhat_lindices = np.tril_indices_from(dhat, k=-1)
        while True:
            b = self.bmat(dhat, wgths, d)
            y = np.dot(v, np.dot(b, x))
            y = x + self.relax*(y-x)
            combinations = np.indices((y.shape[0], y.shape[0]))
            # e = np.zeros((x.shape[0], x.shape[0]))
            # for i in range(y.shape[0]):
            #     for j in range(i+1, y.shape[0]):
            #         e[i, j] = np.sqrt(np.sum(np.square(y[i,:] - y[j,:])))
            #         e[j, i] = e[i, j]
            e = np.sqrt(np.sum(np.square(y[combinations[0], :] - y[combinations[1], :]), axis=2))
            
            dhat2 = self.transform(e, disobj, w=wgths, normq=nn)
            wgths_indices = np.triu_indices_from(wgths, k=1)
            e_indices = np.triu_indices_from(e, k=1)
            snon = np.sum(wgths[wgths_indices]*np.square(dhat2['res']-e[e_indices]))/nn
            
            if self.verbose:
                print(f"Iteration: {itel:3}  Stress (raw): {snon:12.8f}  Difference: {sold-snon:12.8f}")
            
            if ((sold-snon) < self.eps) or (itel == self.itmax):
                break
            x = y
            d = e
            sold = snon
            dhat[dhat_uindices] = dhat2['res']
            dhat[dhat_lindices] = dhat[dhat_lindices[1], dhat_lindices[0]]
            itel = itel+1

        stress = np.sqrt(snon)
        dhat[np.isnan(diss)] = np.nan

        # confdiss = self.normDissN(e, wgths, 1)

        combinations = np.indices((y.shape[0], y.shape[0]))
        dy = np.sqrt(np.sum(np.square(y[combinations[0], :] - y[combinations[1], :]), axis=2))
        spoint = self.spp(dhat, dy, wgths)
        rss = np.sum(spoint['resmat'][np.tril_indices_from(spoint['resmat'], k=-1)])

        if itel == self.itmax:
            print("Iteration limit reached! You may want to increase the itmax argument!")

        return {'delta': diss,
                'dhat': dhat,
                'confdist': dy,
                'iord': dhat2['iord_prim'],
                'conf': y,
                'stress': stress,
                'spp': spoint['spp'],
                'ndim': p,
                'weightmat': wgths,
                'resmat': spoint['resmat'],
                'rss': rss,
                'init': xstart,
                'model': "Symmetric SMACOF",
                'type': self.type}

# with open("G:\Mestrado\BD\data\idiss.csv") as csvfile:
# with open("../data/idiss.csv") as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     array = []
#     for row in spamreader:
#         line = []
#         for item in row:
#             line.append(float(item))
#         array.append(line)
#     idiss = np.array(array)

# # print(idiss)
# # idiss = np.ones((6480, 6480))
# idiss[0, 1] = idiss[1, 0] = np.nan
# idiss[3, 2] = idiss[2, 3] = np.nan
# model = MDS(type = "interval", verbose=False).fit(idiss)
# print(model['conf'])
# [[ 0.70485799 -0.24628322]
#  [ 0.49016144 -0.35819016]
#  [ 0.00525712 -0.58601052]
#  [-0.58801328 -0.20534594]
#  [-0.77183538 -0.10066397]
#  [-0.3669543   0.27672676]
#  [-0.01199711  0.76243705]
#  [ 0.53852353  0.45733001]]