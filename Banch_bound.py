

class Tree:
    def __init__(self, replacements, rate, index_not1, index_not2, mol1, mol2):
        self.replacements = replacements
        self.index_not1 = index_not1
        self.index_not2 = index_not2
        r = rate(mol1, mol2)
        self.list1 = [(r, mol1)]
        self.list2 = [(r, mol2)]
        self.top = [(r, (mol1, mol2))]
        self.rate = rate

    def update_best(self, temp1, temp2, temp_rate):
        temp = (temp1, temp2)
        ind = bisect(self.top, (temp_rate, temp))
        if ind == 0 or self.top[ind - 1] != (temp_rate, temp):
            self.top.insert(ind, (temp_rate, temp))
            if len(self.top) > 100:
                self.top.pop(0)

    def check_index(self, ind, index_not):
        for i in index_not:
            if i[0] <= ind <= i[1]:
                return True
        return False

    def add(self, el):
        self.list.append((self.rate(el), el))

    def step(self):
        lis = []
        kk = randint(0, 1)
        if kk == 0:
            self.list = self.list1[:]
            self.index_not = self.index_not1[:]
        else:
            self.list = self.list2[:]
            self.index_not = self.index_not2[:]
        temp = self.top[-1][-1][0]
        temp2 = self.top[-1][-1][1]
        for base_rate, belok in self.list:
            s = list(belok)
            for i in sample(range(len(s)), k=len(s) // 2):
                for j in sample(self.replacements, k=len(self.replacements) // 2):
                    if j == s[i] or self.check_index(i, self.index_not):
                        continue
                    if kk == 0:
                        temp = ''.join(s[:i] + [j] + s[i + 1:])
                    else:
                        temp2 = ''.join(s[:i] + [j] + s[i + 1:])
                    temp_rate = self.rate(temp, temp2)
                    if temp_rate > base_rate:
                        if kk == 0:
                            ind = bisect(lis, (temp_rate, temp))
                            if ind == 0 or lis[ind - 1] != (temp_rate, temp):
                                lis.insert(ind, (temp_rate, temp))
                                self.update_best(temp, temp2, temp_rate)
                                if len(lis) > 100:
                                    lis.pop(0)
                        else:
                            ind = bisect(lis, (temp_rate, temp2))
                            if ind == 0 or lis[ind - 1] != (temp_rate, temp2):
                                lis.insert(ind, (temp_rate, temp2))
                                self.update_best(temp, temp2, temp_rate)
                                if len(lis) > 1000:
                                    lis.pop(0)
        if kk == 0:
            self.list1 = list(set(lis))
        else:
            self.list2 = list(set(lis))
        print(kk, self.top[-1])

    def perebor(self, k=100):
        for i in range(k):
            self.step()
            if len(self.list) == 0:
                return self.top
        return self.top
