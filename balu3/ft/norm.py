def minmax(X):
    ma = X.max(0)
    mi = X.min(0)
    md    = ma-mi    
    a     = 1/ md
    b     = -mi/md
    return X * a + b, a, b

def mean0(X):
    mf = X.mean(0)
    sf = X.std(0)
    a  = 1/sf;
    b  = -mf/sf;
    return X * a + b, a, b
