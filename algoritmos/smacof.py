import csv
import numpy as np
from numpy import linalg as LA

class MDS:
    def __init__(self, type="ratio", ndim = 2, weight_option = 0, 
                     init = "torgerson", ties = "primary", relax = False, 
                     modulus = 1, itmax = 1000, eps = 1e-6, spline_degree = 2, 
                     spline_intKnots = 2):
        self.type = type
        self.ndim = ndim
        self.weight_option = weight_option
        self.init = init
        self.ties = ties
        self.relax = relax
        self.modulus = modulus
        self.itmax = itmax
        self.eps = eps
        self.spline_degree = spline_degree
        self.spline_intKnots = spline_intKnots
    
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
        d_indices = np.triu_indices_from(d, k=1)
        d[5, 0] = d[0, 5] = 1e-13
        z = np.where(d < eps, 1, 0)
        b = (wgths*diss*(1-z))/(d+z)
        r = np.sum(b, axis=1)
        return np.diag(r)-b

    def smacof(self, delta):
        diss = delta
        p = self.ndim
        n = diss.shape[0]
        if (p > (n - 1)):
            raise TypeError(f"Maximum number of dimensions is {n-1}!")

        nn = n*(n-1)/2
        opcao = ['um', 'um_ou_zero', '1/d', '1/d2']
        opcao_n = self.weight_option
        if opcao[opcao_n] == 'um':
            wgths = np.ones_like(diss)
        elif opcao[opcao_n] == 'um_ou_zero':
            wgths = np.where(diss == np.max(diss), 0., 1.)
        elif opcao[opcao_n] == '1/d':
            diss = np.where(diss == 0., np.inf, diss)
            d_min = np.min(diss)
            wgths = (1. / diss) / (1. / d_min)
            diss = np.where(diss == np.inf, 0, diss)
        elif opcao[opcao_n] == '1/d2':
            diss = np.where(diss == 0., np.inf, diss)
            d_min = np.min(diss)
            wgths = (1. / diss ** 2) / (1. / d_min ** 2)
            diss = np.where(diss == np.inf, 0, diss)
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
        d = np.sqrt(np.sum(np.square(x[combinations[0], :] - x[combinations[1], :]), axis=2))
        lb = np.nansum(wgths*d*dhat)/np.nansum(wgths*np.square(d))
        x = lb*x
        d = lb*d

        sold = np.nansum(wgths*np.square(dhat-d))/(2*nn)
        while True:
            b = self.bmat(dhat, wgths, d)
            y = np.dot(v, np.dot(b, x))
            y = x + self.relax*(y-x)
            combinations = np.indices((y.shape[0], y.shape[0]))
            e = np.sqrt(np.sum(np.square(y[combinations[0], :] - y[combinations[1], :]), axis=2))
            ssma = np.sum(wgths*np.square(dhat-e))/2

            break

        # tirar break do original
        return 0.1

with open("G:\Mestrado\BD\data\idiss.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    array = []
    for row in spamreader:
        line = []
        for item in row:
            line.append(float(item))
        array.append(line)
    idiss = np.array(array)

idiss[0, 1] = idiss[1, 0] = np.nan
idiss[3, 2] = idiss[2, 3] = np.nan
model = MDS().smacof(idiss)