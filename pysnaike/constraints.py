import numpy as np

class MaxNorm():
    def __init__(self, max_val):
        self.max_val = max_val

    def constrain(self, matrix):
        if len(matrix.shape) == 1:
            norm = np.linalg.norm(matrix, ord=np.inf)
            if norm > self.max_val:
                matrix = matrix / norm * self.max_val        
        else:
            normalized = np.linalg.norm(matrix, ord=np.inf, axis=len(matrix.shape) -1)
            smaller = np.where(normalized <= self.max_val)
            greater = np.where(normalized > self.max_val)
            normalized[smaller] = 1            
            matrix = (matrix.T * (1/normalized.T)).T
            matrix[greater] *= self.max_val
        return matrix


class NoNorm():
    def constrain(self, matrix):        
        return matrix

def no_norm():
    return NoNorm()    

def max_norm(constraint):
    return MaxNorm(constraint)    
