CubeVortex = [[3, 3, 0],
              [0, 3, 0],
              [0, 3, 3],
              [3, 3, 3],
              [0, 0, 3],
              [3, 0, 3],
              [3, 0, 0],
              [0, 0, 0]]

a = np.array([[5, 1, 3],
              [1, 1, 1],
              [1, 2, 1]])
b = np.array([1, 2, 3])
print a.dot(b)
# array([16, 6, 8])
