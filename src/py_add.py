# This is not how you write python

def py_add(N):
    x, y = [], []
    for i in range(N):
        x.append(float(1))
        y.append(float(2))

    for i in range(N):
        y[i] = x[i] + y[i]

    maxError = float(0)
    for i in range(N):
        maxError = abs(y[i] - 3) if abs(y[i] - 3) > maxError else maxError

    return maxError


if __name__ == '__main__':
    print(py_add(1<<20))
