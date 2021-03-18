import numpy as np
from tqdm import tqdm


def random_search(n, dims):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens = np.zeros((n, dims))

    for i, gen in enumerate(gens):
        r = np.random.randint(1, dims)
        for _r in range(r):
            gen[_r] = 1
        np.random.shuffle(gen)
    return gens


def MyCost(x):
    return np.sum(x)

def ind2vec(ind, N=None):
  ind = np.asarray(ind)
  if N is None: 
      N = ind.max() + 1
  return (np.arange(N) == ind[:,None]).astype(int)

def vec2ind(vec):
  vec = np.asarray(vec)
  return ind2vec(np.array([1,2,3,4])).nonzero()[1].astype(int)


def BDA(N, max_iter, nVar, CostFunction):
    dim = nVar

    food_fit = float("-inf")
    food_pos = np.zeros((dim, 1))

    enemy_fit = float("-inf")
    enemy_pos = np.zeros((dim, 1))

    X = random_search(N, dim)
    DeltaX = random_search(N, dim)

    fit = np.zeros(N)

    for iter in tqdm(range(max_iter), desc="Training..."):
        w = 0.9 - iter * ((0.9-0.4) / max_iter)
        m_c = 0.1 - iter * ((0.1-0) / (max_iter/2))

        if m_c < 0:
            m_c = 0

        s = 2 * np.random.randn() * m_c
        a = 2 * np.random.randn() * m_c
        c = 2 * np.random.randn() * m_c
        f = 2 * np.random.randn()
        e = m_c

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

            Neighbours_DeltaX = np.zeros((N, dim))
            Neighbours_X = np.zeros((N, dim))

            for j in range(N):
                if (i != j):
                    index += 1
                    neighbours_no += 1
                    Neighbours_DeltaX[index] = DeltaX[j]
                    Neighbours_X[index] = X[j]

            # Separation=====================================================================================
            # Eq. (3.1)
            S = np.zeros(dim)
            for k in range(neighbours_no):
                S = S + (Neighbours_X[k] - X[i])
            S = -S

            # Separation=====================================================================================
            # Eq. (3.2)
            # np.sum(Neighbours_DeltaX)/neighbours_no
            A = np.array([sum([_[_d] for _ in Neighbours_DeltaX]) /
                         neighbours_no for _d in range(dim)])

            # Separation=====================================================================================
            # Eq. (3.3)
            C = np.array([_-g for _, g in zip([sum([_[_d]
                         for _ in Neighbours_DeltaX])/neighbours_no for _d in range(dim)], X[i])])

            # Separation=====================================================================================
            # Eq. (3.4)
            F = np.array([fp-g for fp, g in zip(food_pos, X[i])])

            # Separation=====================================================================================
            # Eq. (3.5)
            E = np.array([ep+g for ep, g in zip(enemy_pos, X[i])])

            # print("Sizes", S.shape, A.shape, C.shape, F.shape, F.shape, DeltaX.shape)
            for j in range(dim):
                # Eq. (3.6)

                DeltaX[i, j] = s*S[j] + a*A[j] + c * \
                    C[j] + f*F[j] + e*E[j] + w*DeltaX[i, j]

                if DeltaX[i, j] > 6:
                    DeltaX[i, j] = 6
                if DeltaX[i, j] < -6:
                    DeltaX[i, j] = -6

                T = abs(DeltaX[i, j] / np.sqrt((1+DeltaX[i, j]**2)))
                if np.random.randn() < T:
                    X[i, j] = 1 if X[i, j] == 0 else 0

    # Convergence_curve(iter)=food_fit
    Best_pos = food_pos
    Best_score = food_fit
    best_dict = {'Best_pos': Best_pos, 'Best_score': Best_score}
    return best_dict


if __name__ == "__main__":
    Max_iteration = 500
    N = 60
    nVar = 40

    BDA(N, Max_iteration, nVar, MyCost)
