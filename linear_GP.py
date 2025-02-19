import random
import numpy as np
import sympy
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import math
from settings import *
matplotlib.use('TkAgg')

RANGE           = 10    # 式の長さ ※RANGE * 2 + 1 となるので21
MAX_RANGE       = 10*RANGE   # 最大の式の長さ
MIN_RANGE       = RANGE     # 最小の式の長さ

#def target_func(x):
#    return  x ** 3 + 2 * x ** 2 + 3 * x + 4

FUNCTIONS = ['+', '-', '*']
TERMINALS = ['x', -2, -1, 0, 1, 2]

def main():
    random.seed()
    list = make_list() #初期個体作成
    i = 0
    fit_list = [1]
    min_fit = []
    avg_fit = []
    while min(fit_list) > 0.01 and i < GENERATIONS:
        list, fit_list = fitness(list) #適応度計算
        if random.random() < XO_RATE:
            list = crossover(list)
            list, fit_list = fitness(list, SE_WAY) #再度適応度計算 ここで適応度下位の10個を切り捨てたい
        if random.random() < PROB_MUTATION:
            list = mutation(list)
        i += 1
        #個体の長さの平均
        s = 0
        for j in range(len(list)):
            s += len(list[j])
        print("最良適応度", min(fit_list), "平均適応度", float(sum(fit_list) / len(fit_list)))
        min_fit.append(min(fit_list))
        avg_fit.append(float(sum(fit_list) / len(fit_list)))
        print("世代数", i, "平均個体長", s / POP_SIZE)
        display(list, fit_list, 0)
    n = fit_list.index(min(fit_list))
    print("最良個体", list[n], "適応度", fit_list[n])
    for i in range(len(min_fit)):
        print(min_fit[i])
    for i in range(len(avg_fit)):
        print(avg_fit[i])    
        
    display(list, fit_list, 1)

def make_list():
    list1 = []
    for i in range(POP_SIZE):
        list2 = []
        term = TERMINALS[int(random.randint(0, len(TERMINALS)-1))]
        list2.append(term)
        for i in range(RANGE):
            func = FUNCTIONS[int(random.randint(0, len(FUNCTIONS)-1))]
            term = TERMINALS[int(random.randint(0, len(TERMINALS)-1))]
            list2.append(func)
            list2.append(term)
        list1.append(list2)
    return list1

def fitness(list, n="none"):
    form = []
    p = []
    for i in range(-10, 11):
        p.append(i)
    for i in range(len(list)):
        form.append('')
        for j in range(len(list[i])):
            if str(list[i][j]) == 'x':
                form[i] += 'x'
            else:
                form[i] += str(list[i][j])
    x = sympy.Symbol('x')
    fit_list = []
    j = 0
    for i in form:
        expr = sympy.sympify(i)
        fit_list.append(0)
        for k in p:
            fit_list[j] += (target_func(k) - expr.subs(x, k)) ** 2 # 二乗誤差
            #fit_list[j] += (1 /((abs(target_func(k) - expr.subs(x, k))) + 1) )/ len(p)
        j += 1

    #適応度が低いn個を削除 エリート選択
    if n == "elite":
        while len(list) > POP_SIZE:
            max1 = max(fit_list)
            max1_i = fit_list.index(max1)
            list.pop(max1_i)
            fit_list.pop(max1_i)
    #ルーレット選択(完成はしたがうまく動かない？)
    if n == "roulette":
        list, fit_list = roulette(list, fit_list)            

    return list, fit_list

def roulette(list, fit_list):
    while len(list) > POP_SIZE:
        fit_list_r = []
        for i in range(len(fit_list)):
            fit_list_r.append(0)
            fit_list_r[i] = fit_list[i]
        sum_fit_r = sum(fit_list_r)
        for i in range(len(fit_list_r)):
            fit_list_r[i] = float(fit_list_r[i] / sum_fit_r)        
        for i in range(1, len(fit_list_r)):
            fit_list_r[i] += fit_list_r[i - 1]
        r = random.random() 
        for i in fit_list_r:
            if r <= i:
                list.pop(fit_list_r.index(i))
                fit_list.pop(fit_list_r.index(i))
                break
    return list, fit_list
        
def crossover(list):
    #sum_fit = sum(fit_list)
    #select_number = random.choices(range(len(list)), weights=fit_list, k=2)
    select_number = random.sample(range(len(list)), 2)
    parent1 = list[select_number[0]]
    parent2 = list[select_number[1]]
    n = 0
    if XO_WAY == 2: #二点交叉
        while n < ADD_CHILD:
            r1 = r2 = 0
            while r1 == r2 or r1 > r2:
                r1 = random.randrange(0, len(parent1), 2)
                r2 = random.randrange(0, len(parent1), 2)
            r3 = r4 = 0
            while r3 == r4 or r3 > r4:
                r3 = random.randrange(0, len(parent2), 2)
                r4 = random.randrange(0, len(parent2), 2)
            child1 = parent1[:r1] + parent2[r3:r4] + parent1[r2:]
            child2 = parent2[:r3] + parent1[r1:r2] + parent2[r4:]
            if len(child1) > MAX_RANGE or len(child2) > MAX_RANGE or len(child1) < MIN_RANGE or len(child2) < MIN_RANGE:
                continue
            list.append(child1)
            list.append(child2)
            n += 2
    if XO_WAY == 1: #一点交叉
        while n < ADD_CHILD:
            cut = random.randint(0, 1) #偶奇選択
            r1 = random.randrange(cut, len(parent1), 2)
            r2= random.randrange(cut, len(parent2), 2)
            child1 = parent1[:r1] + parent2[r2:]
            child2 = parent2[:r2] + parent1[r1:]
            if len(child1) > MAX_RANGE or len(child2) > MAX_RANGE or len(child1) < MIN_RANGE or len(child2) < MIN_RANGE:
                continue
            list.append(child1)
            list.append(child2)
            n += 2
    return list            
    
def mutation(list):
    n = random.randrange(0, len(list))
    m = random.randrange(0, len(list[n]))
    if m % 2 == 0:
        list[n][m] = TERMINALS[int(random.randint(0, len(TERMINALS)-1))]
    else:
        list[n][m] = FUNCTIONS[int(random.randint(0, len(FUNCTIONS)-1))]
    return list

def display(list, fit_list, n):
    x = np.linspace(-10, 10, 100)
    
    min_f = min(fit_list)
    min_f_i = fit_list.index(min_f)
    form = ''
    for i in range(len(list[min_f_i])):
        if str(list[min_f_i][i]) == 'x':
            form += 'x'
        else:
            form += str(list[min_f_i][i])
    

    y1 = eval(form)
    plt.plot(x, y1, label="Fitted Function")
        
    y2 = target_func(x)
    plt.plot(x, y2, label="Target Function")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
  
    if n == 0:
        plt.pause(0.0001)
        plt.clf()
    if n == 1:
        plt.show()
    
#main()
        