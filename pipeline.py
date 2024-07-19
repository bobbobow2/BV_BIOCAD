from bisect import bisect
from random import randint, sample, choice, choices
from time import sleep
from utils.general import FVbinder, CustomIgFold
from utils.scorer import loss, get_losses
import warnings
import os
import json
warnings.filterwarnings("ignore")
from antiberty import AntiBERTyRunner


antiberty = AntiBERTyRunner()
# ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


sc = CustomIgFold()


class GenAlgorythm:
    def __init__(self, rate, mutate, generate_popul, binder, k_mutation=10):
        sc.fold(binder)
        self.rasmetka_l = [i[0] for i in get_rasmetka(binder.vl.seq)]
        self.rasmetka_h = [i[0] for i in get_rasmetka(binder.vh.seq)]
        with open('HH.json', 'r') as f:
            HH = json.loads(f.read())
        with open('LL.json', 'r') as f:
            LL = json.loads(f.read())
        self.replacements_h = [list(HH[i].keys()) for i in self.rasmetka_h if i in HH]
        self.replacements_l = [list(LL[i].keys()) for i in self.rasmetka_l if i in LL]
        self.weights_h = [list(HH[i].values()) for i in self.rasmetka_h if i in HH]
        self.weights_l = [list(LL[i].values()) for i in self.rasmetka_l if i in LL]
        r = rate(binder)
        print(r)
        self.list = [(r, binder)]
        self.top = [(r, binder)]
        self.rate = rate
        self.mutate = mutate
        self.k_mutation = k_mutation
        generate_popul(self)
        self.list.sort(key=lambda x: x[0])

    def update_best(self, temp, temp_rate):
        ind = bisect(self.top, (temp_rate, temp))
        if ind == 0 or self.top[ind - 1] != (temp_rate, temp):
            self.top.insert(ind, (temp_rate, temp))
            if len(self.top) > 100:
                self.top.pop(0)

    def add(self, vl, vh=None):
        if vh is None:
            vl, vh = vl
        binder = FVbinder(vh=vh, vl=vl)
        sc.fold(binder)
        self.list.append((self.rate(binder), binder))

    def add_binder(self, binder):
        sc.fold(binder)
        self.list.append((self.rate(binder), binder))

    def popul(self, binder1, binder2):
        vl1 = list(binder1.vl.seq)
        vl2 = list(binder2.vl.seq)
        vh1 = list(binder1.vh.seq)
        vh2 = list(binder2.vh.seq)
        indl = randint(len(vl1) // 10, len(vl1) // 10 * 9)
        indh = randint(len(vh1) // 10, len(vh1) // 10 * 9)
        sl = vl1[:indl] + vl2[indl:]
        # sl = self.mutate(self, sl, binder1.vl_cdr_mask, 'L')
        sh = vh1[:indh] + vh2[indh:]
        # sh = self.mutate(self, sh, binder1.vh_cdr_mask, 'H')

        sh, sl = mutate_bert(self, sl, sh, binder1.vl_cdr_mask, binder1.vh_cdr_mask)
        sh, sl = ''.join(sh), ''.join(sl)
        
        binder = FVbinder(sh, sl, None, binder1.vh_cdr_mask, binder1.vl_cdr_mask)
        sc.fold(binder)
        return binder

    def step(self):
        lis = []
        par = self.list[:]
        for i in range(50):
            seqs = sample(par, k=2)
            binder1 = seqs[0][1]
            binder2 = seqs[1][1]
            binder = self.popul(binder1, binder2)

            r = self.rate(binder)
            ind = bisect(self.list, (r, binder))
            self.list.insert(ind, (r, binder))
            self.update_best(binder, r)
        for r, binder in par:
            ind = bisect(self.list, (r, binder))
            self.list.insert(ind, (r, binder))
            self.update_best(binder, r)
        self.list = self.list[-100:]

    def perebor(self, k=20):
        for i in range(k):
            self.step()
            print(f'[LOG] step {i + 1}: {self.top[-1]}')
        return self.top


def pipeline(mol1, mol2, tar1, tar2):
    popul = FVbinder(vl=mol1, vh=mol2)
    global tar
    tar = FVbinder(vl=tar1, vh=tar2)
    sc.fold(tar)
    t = GenAlgorythm(rate, mutate, generate_popul, popul)
    answer = t.perebor()[-1]
    return rate(popul), get_losses(popul, tar), answer, get_losses(answer[1], tar)


def rate2(s1, s2):
    try:
        p2 = sc.fold(s2, s1, 'e2')
        print('[LOG] done')
        return sc.rmsd_score(p1, p2)
    except Exception as e:
        print('[ERROR]', e)
        return -100000000


def rate3(s1, s2):
    s = ('H', 'R', 'K')
    k = 0
    for i in s1:
        if i in s:
            k += 1
    for i in s2:
        if i in s:
            k += 1
    return k


def rate(binder):
    # print(binder)
    global tar
    return loss(binder, tar)


def mutate(gen, s, cdr_mask, type='H'):
    s = list(s)
    try:
        if type == 'L':
            return mutate_l(gen, s, cdr_mask)
        return mutate_h(gen, s, cdr_mask)
    except:
        return ''.join(s)


def mutate_bert(gen, vl, vh, cdr_mask_l, cdr_mask_h):
    for i in sample(range(len(vl)), k=randint(0, gen.k_mutation)):
        if cdr_mask_l[i].item():
            continue
        vl[i] = '_'
    for i in sample(range(len(vh)), k=randint(0, gen.k_mutation)):
        if cdr_mask_h[i].item():
            continue
        vh[i] = '_'
    sequences = [vh, vl]
    return antiberty.fill_masks(sequences) 


def generate_popul(gen):
    k_old = gen.k_mutation
    gen.k_mutation = 10
    for i in range(99):
        gen.add_binder(gen.popul(gen.top[-1][-1], gen.top[-1][-1]))
    gen.k_mutation = k_old


def mutate_l(gen, s, cdr_mask):
    for i in sample(range(len(s)), k=randint(0, gen.k_mutation)):
        j = choices(gen.replacements_l[i], gen.weights_l[i])[0]
        if j == s[i] or cdr_mask[i].item():
            continue
        s[i] = j
    return s


def mutate_h(gen, s, cdr_mask):
    for i in sample(range(len(s)), k=randint(0, gen.k_mutation)):
        j = choices(gen.replacements_h[i], gen.weights_h[i])[0]
        if j == s[i] or cdr_mask[i].item():
            continue
        s[i] = j
    return s


def get_rasmetka(mol):
    os.system(f'ANARCI -i {mol} --scheme chothia --outfile temp1.fasta')
    with open('temp1.fasta', 'r') as f:
        data1 = f.readlines()[7:-1]
    s1 = []
    for i in data1:
        i = i.rstrip('\n')[2:]
        s1.append((i[:-1], i[-1]))
    return s1
