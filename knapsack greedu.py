
p = [60,100,120]
w = [10,20,30]
wt = 50
for i in range(len(p)):
    for j in range(0,len(p)):
        if(i!=j and i<j):
            if((w[i]+w[j])==wt):
                print(p[i]+p[j])