first = 0
second = 1
n = 10 
print(first)
print(second)

for i in range(1, n):
    third = first+second
    first,second = second,third
    print(third)

