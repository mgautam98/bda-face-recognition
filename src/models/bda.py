import numpy as np

def random_search(n,dims):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens = np.zeros((n, dims))

    for i,gen in enumerate(gens) :
        r = np.random.randint(1,dims)
        for _r in range(r):
            gen[_r] = 1
        random.shuffle(gen)
    return gens

def BDA(N, max_iter, nVar, CostFunction):
  dim = nVar
  
  food_fit = float("-inf")
  food_pos = np.zeros((1,dim))

  enemy_fit = float("-inf")
  enemy_pos = np.zeros((1,dim))
  
  fit = np.zeros((1,N))
  X = random_search(N,dim)
  DeltaX = random_search(N,dim)


  for iter in range(max_iter):
    w = 0.9 - iter * ((0.9-0.4) / max_iter)
    mc = 0.1- iter * ((0.1-0) / (max_iter/2))

    if mc < 0:
      mc=0

    s = 2 * np.random.randn() * mc
    a = 2 * np.random.randn() * mc
    c = 2 * np.random.randn() * mc
    f = 2 * np.random.randn()
    e = mc

    if iter > (3 * max_iter / 3):
      e = 0
    
    for i in range(N):
      fit[i] = CostFunction(X[i])

      if fit[i] < food_fit:
        food_fit = fit[i]
        food_pos = X[i]
      
      if fit[i] > enemy_fit:
        enemy_fit = fit[i]
        enemy_pos = X[i]
    
    for i in range(N):
      index = -1
      neighbours_no = -1

      Neighbours_DeltaX = np.zeros((N,dim)) 
      Neighbours_X = np.zeros((N,dim)) 

      for j in range(N):
        if (i != j):
          index += 1
          neighbours_no += 1
          Neighbours_DeltaX[index] = DeltaX[j]
          Neighbours_X[index] = X[j]
      
      # Separation=====================================================================================
      # Eq. (3.1)
      S=np.zeros((dim,1))
      for k in range(neighbours_no):
        S = S + (Neighbours_X[k] - X[i])
      S = -S

      # Separation===================================================================================== 
      # Eq. (3.2)
      A = [sum([_[_d] for _ in Neighbours_DeltaX])/neighbours_no for _d in range(dim)] #np.sum(Neighbours_DeltaX)/neighbours_no

      # Separation===================================================================================== 
      # Eq. (3.3)
      C = [_-g for _,g in zip([sum([_[_d] for _ in Neighbours_DeltaX])/neighbours_no for _d in range(dim)],X[i])]


      # Separation===================================================================================== 
      # Eq. (3.4)
      F = [fp-g for fp,g in zip(food_pos,X[i])]

      # Separation===================================================================================== 
      # Eq. (3.5)
      E = [ep+g for  ep,g in zip(enemy_pos,X[i])]

      for j in range(dim):
        # Eq. (3.6)
        DeltaX[i][j] = s*S[j] + a*A[j] + c*C[j] + f*F[j] + e*E[j] + w*DeltaX[i][j]

        if DeltaX[i][j] > 6:
          DeltaX[i][j] = 6
        if DeltaX[i][j] < -6:
          DeltaX[i][j] = -6

        T = abs(DeltaX[i][j] / math.sqrt((1+DeltaX[i][j]**2)))
        if np.random.randn() < T:
          X[i][j]=1 if X[i][j] == 0 else 0
        
  # Convergence_curve(iter)=food_fit
  Best_pos=food_pos
  Best_score=food_fit
  best_dict={'Best_pos':Best_pos , 'Best_score':Best_score}
  return best_dict

def MyCost(x):
  return np.sum(x)

if __name__ == "__main__":
  Max_iteration = 500
  N = 60
  nVar = 40

  BDA(N, Max_iteration, nVar, MyCost)

