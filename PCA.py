import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

class PCA:

    def __init__(self, Data):
        self.Data = Data
        self.SetSize = Data.shape[1] # training set size
        self.NormData, self.RowMean, self.MagMean = self.NormData()

    def NormData(self):
        #mean normalization. feature scaling not necessary
        RowMean = np.zeros((self.Data.shape[0],))
        MagMean = 0
        for x in range(self.SetSize):
            RowMean += self.Data[:,x]
            MagMean += np.linalg.norm(self.Data[:,x])
        RowMean /= self.SetSize
        MagMean /= self.SetSize
        NormData = np.empty_like(self.Data)
        for x in range(self.SetSize):
            NormData[:,x] = self.Data[:,x]-RowMean
        return NormData, RowMean, MagMean

    def ProcessData(self, k):
        AverageMatrix = np.zeros((self.Data.shape[0],self.Data.shape[0]))
        #comuting covariance matrix to reduce from n dimensions to k dimensions
        #n = rows = pixels
        print(self.SetSize)
        for x in range(self.SetSize):
            column_vector = self.NormData[:,x].reshape(self.NormData.shape[0],1)
            AverageMatrix += column_vector.dot(column_vector.T)
        AverageMatrix = AverageMatrix/self.SetSize
        (w,v) = eigsh(AverageMatrix, k=k)
 #       (w,v) = np.linalg.eig(AverageMatrix)
 
        print("check3")
        v_reduce = v[:,:k] #taking the first k columns of the Self.SetSize length eigenvectors. Taking all rows from columns 0 to k.
        z = v_reduce.T.dot(self.NormData) #input can be any vector (face) that we want to plot along PCs
        #v_reduce = eigenfaces, z = coefficients
        print(v_reduce)
        return v_reduce, z
        

if __name__ == "__main__":
    rand = np.random.random((400,14))
    
    #Waysek = InputConversion("Waysek1.jpg")

    p = PCA(rand)
    print(rand[4,4])
    print (p.ProcessData(7))
        
        
