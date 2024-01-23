N = 8

A = [[x] for x in range(N)]

# iteration 1:
i = 1
A1 = [[A[(n + n//(N//2**i)) % 8], A[(n + n//(N//2**i) + N//2**i) % 8]] for n in range(N)]
print(A1)

i = 2
A2 = [[A1[(n + n//(N//2**i)) % 8], A1[(n + n//(N//2**i) + N//2**i) % 8]] for n in range(N)]
print(A2)

i = 3
A3 = [[A2[(n + n//(N//2**i)) % 8], A2[(n + n//(N//2**i) + N//2**i) % 8]] for n in range(N)]
print(A3)
