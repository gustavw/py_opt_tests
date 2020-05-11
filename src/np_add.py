import numpy as np

def np_add(N):
    x = np.full((N, 1), np.float64(1))
    y = np.full((N, 1), np.float64(2))
    y = np.add(x, y)

    maxError = np.max(
        np.abs(
            np.subtract(y, np.full((N, 1), np.float64(3)))
        )
    )

    return maxError

if __name__ == '__main__':
    print(np_add(1<<20))