from bisect import bisect
from multiprocessing import Pool
from random import randint, sample, choice
from time import sleep


class GenAlgorythm:
    def __init__(self, replacements, rate, index_not1, index_not2, mol1, mol2, k_mutation=2):
        self.replacements = replacements
        self.index_not1 = index_not1
        self.index_not2 = index_not2
        r = rate(mol1, mol2)
        self.list1 = [(r, mol1)]
        self.list2 = [(r, mol2)]
        self.top = [(r, (mol1, mol2))]
        self.rate = rate
        self.k_mutation = k_mutation * 40
        for i in range(100):
            self.add(self.popul(self.top[-1][-1][0], self.top[-1][-1][0], index_not1), 0)
        for i in range(100):
            self.add(self.popul(self.top[-1][-1][1], self.top[-1][-1][1], index_not2), 1)
        self.k_mutation //= 40
        self.list1.sort()
        self.list2.sort()

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

    def add(self, el, kk):
        if kk == 0:
            self.list1.append((self.rate(el, self.top[-1][-1][1]), el))
        else:
            self.list2.append((self.rate(self.top[-1][-1][0], el), el))

    def popul(self, r1, r2, index_not):
        r1 = list(r1)
        r2 = list(r2)
        ind = randint(len(r1) // 10, len(r1) // 10 * 9)
        s = r1[:ind] + r2[ind:]
        for i in sample(range(len(s)), k=self.k_mutation):
            j = choice(self.replacements)
            if j == s[i] or self.check_index(i, index_not):
                continue
            s[i] = j
        return ''.join(s)

    def step(self):
        lis = []
        kk = randint(0, 1)
        if kk == 0:
            self.list = self.list1
            index_not = self.index_not1[:]
        else:
            self.list = self.list2
            index_not = self.index_not2[:]
        temp = self.top[-1][-1][0]
        temp2 = self.top[-1][-1][1]
        par = self.list[:]
        for i in range(20):
            rrrr = sample(par, k=2)
            r1 = rrrr[0][1]
            r2 = rrrr[1][1]
            popul = self.popul(r1, r2, index_not)
            if kk == 0:
                r = self.rate(popul, temp2)
                ind = bisect(self.list, (r, popul))
                self.list.insert(ind, (r, popul))
                self.update_best(popul, temp2, r)
            else:
                r = self.rate(temp, popul)
                ind = bisect(self.list, (r, popul))
                self.list.insert(ind, (r, popul))
                self.update_best(temp, popul, r)
        for popul in par:
            ind = bisect(self.list, popul)
            self.list.insert(ind, popul)
            self.update_best(popul[1], temp2, popul[0])
        if kk == 0:
            self.list1 = self.list[-100:]
        else:
            self.list2 = self.list[-100:]
        # print(kk, self.top[-1])

    def perebor(self, k=100):
        for i in range(k):
            self.step()
            if len(self.list) == 0:
                return self.top
        return self.top


def rate(pos, pos2):
    s = ('H', 'R', 'K')
    k = 0
    for i in pos:
        if i in s:
            k += 1
    for i in pos2:
        if i in s:
            k += 1
    return k


mol1 = 'QSVLTQPPSVSEAPRQRVTISCSGSSSNIGNNAVNWYQQLPGKAPKLLIYYDDLLPSGVSDRFSGSKSGTSASLAISGLQSEDEADYYCAAWDDSLNVVVFGGGTKLTVL'
mol2 = 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARAGAVAASKGYYYYYYGMDVWGQGTTVTVSS'
ind1 = [(25, 33), (50, 53), (89, 100)]
ind2 = [(25, 33), (50, 58), (96, 117)]
t = GenAlgorythm(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'], rate, ind1, ind2, mol1, mol2)
