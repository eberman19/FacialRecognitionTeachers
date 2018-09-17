import numpy as np

class PCA:

    def __init__(self, Data):
        self.Data = Data
        self.SetSize = Data.shape[1] # training set size
        self.NormData = self.NormData()

    def NormData(self):
        #mean normalization. feature scaling not necessary
        RowMean = np.zeros((self.Data.shape[0],))
        for x in range(self.SetSize):
            RowMean += self.Data[:,x]
        RowMean /= self.SetSize
        NormData = np.empty_like(self.Data)
        for x in range(self.SetSize):
            NormData[:,x] = self.Data[:,x]-RowMean
        return NormData

    def ProcessData(self, k):
        AverageMatrix = np.zeros((Data.shape[0],Data.shape[0]))
        #comuting covariance matrix to reduce from n dimensions to k dimensions
        #n = rows = pixels
        for x in range(self.SetSize):
            AverageMatrix += self.NormData[:,x].dot(self.NormData[:,x].T)
        AverageMatrix = AverageMatrix/self.SetSize
        (w,v) = np.linalg.eig(AverageMatrix)
        v_reduce = v[:,:k] #taking the first k columns of the Self.SetSize length eigenvectors. Taking all rows from columns 0 to k.
        z = v_reduce.T.dot(self.NormData) #input can be any vector (face) that we want to plot along PCs
        return v_reduce, z
        

if __name__ == "__main__":
    rand = np.random.random((20,5))

    p = PCA(rand)
    print (rand)
    print (p.ProcessData())
        
        
