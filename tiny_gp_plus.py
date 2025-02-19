from statistics import mean
from copy import deepcopy
import graphviz
import matplotlib.pyplot as plt
import random
import sys
import math
from settings import *

MIN_DEPTH       = 2    # 初期の最小深さ
MAX_DEPTH       = 5    # 初期の最大深さ
TOURNAMENT_SIZE = 5   # トーナメント選択のsize

# 真の関数
#def target_func(x):
#    return x ** 3 + 2 * x ** 2 + 3 * x + 4

# 使用できる関数
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]

# nodeに使用できる記号 ここに追加
TERMINALS = ['x', -2, -1, 0, 1, 2]

def main():
    # 引数なしで seed()すると、乱数が都度変化
    random.seed()
    # 真の関数で data点生成
    target_dataset = make_target_dataset()
    # 個体群(木)生成
    population= init_population(POP_SIZE, MAX_DEPTH)
    
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0

    fitnesses = []
    for i in range(POP_SIZE):
        fitnesses.append( fitness(population[i], target_dataset) )
        
    average_f = []
    max_f = []
    gen_l = []

    for gen in range(GENERATIONS):
        gen_l.append(gen + 1)
        nextgen_population=[]
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)  # 交叉
            parent1.mutation()          # 突然変異
            nextgen_population.append(parent1)
        population=nextgen_population

        fitnesses = []
        for i in range(POP_SIZE):
            fitnesses.append( fitness(population[i], target_dataset) )
            
        average_f.append(sum(fitnesses) / len(fitnesses))# for figure
        max_f.append(max(fitnesses))# for figure
        
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("gen:", gen,
                  "best_of_run_f:", round(max(fitnesses),3))
            best_of_run.print_tree()
            
            # best_of_run.draw_tree(
            #     "gen:", gen,
            #     "best_of_run_f:", round(max(fitnesses),3) )
            
        if best_of_run_f == 1: break
    
    print("END OF RUN")
    print("best_of_run attained at gen:", best_of_run_gen,
          " fitness:", round(best_of_run_f,3) )
    
    best_of_run.print_tree()
    best_of_run.draw_tree(
        "best_of_run",
        "gen:" + str(best_of_run_gen) + \
        "has f:"+ str(round(best_of_run_f,3)) )

    print("len(gen_l):",len(gen_l),"len(max_f):",len(max_f) )
    
    show_evolve_graph(gen_l,max_f,average_f)
    
    
def show_evolve_graph(gen_l,max_f,average_f):
    plt.plot(gen_l, max_f,    label = "MAX")
    plt.plot(gen_l, average_f,label = "AVERAGE")
    plt.ylim(0,1)
    plt.xlim(0,)
    plt.title('FITNESS')
    plt.xlabel("EPOCH")
    plt.ylabel("VALUE")
    plt.legend()
    
    plt.show()
    
    
# 真の関数で 101個のdata点生成
def make_target_dataset():
    dataset = []
    for x in range(-100,101,2):
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset

class GPTree:
    def __init__(self):
        self.operator  = None
        self.left  = None
        self.right = None
        
    def node_label(self): # string label
        if self.operator in FUNCTIONS :
            # __name__ には add,sub,mulが入ります
            return self.operator.__name__

        return str(self.operator)
    
    def print_tree(self, prefix = ""):
        print("%s%s" % (prefix, self.node_label()) )
        if self.left:
            self.left.print_tree (prefix + "   ")
        if self.right:
            self.right.print_tree(prefix + "   ")

    def draw_tree(self, fname, footer):
        dot = [graphviz.Digraph(format='svg', filename=fname)]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        dot[0].view()
        
    def draw(self, dot, count):
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)

    # 木にある x に値を代入し、算出
    def compute_tree(self, x): 
        if self.operator in FUNCTIONS:
            return self.operator( self.left.compute_tree(x),
                              self.right.compute_tree(x) )
        elif self.operator == 'x':
            return x
        #ここに関数を追加
        elif self.operator == 'sin(x)':
            return math.sin(x)
        elif self.operator == 'cos(x)':
            return math.cos(x)
        elif self.operator == 'tan(x)':
            return math.tan(x)
        elif self.operator == 'exp(x)':
            return math.exp(x)
        else:
            return self.operator

    # create random tree using either grow or full method
    def random_tree(self, grow, max_depth, depth = 0):
        if depth < MIN_DEPTH or (depth < max_depth and not grow): #深さがMINより小さいor(深さがMAXより小さい and glowがFalse)とき
            self.operator = FUNCTIONS[random.randint(0, len(FUNCTIONS)-1)] #関数をランダムで選ぶ
        elif depth >= max_depth: #深さがMAX以上だったら
            self.operator = TERMINALS[random.randint(0, len(TERMINALS)-1)] #ノードをランダムで選ぶ
        else: # intermediate depth, grow 深さがMAXより小さい and glowがTrue
            if random.random() > 0.5: #1/2の確率で
                self.operator = TERMINALS[random.randint(0, len(TERMINALS)-1)] #ノードをランダムで選ぶ
            else:
                self.operator = FUNCTIONS[random.randint(0, len(FUNCTIONS)-1)] #関数をランダムで選ぶ
        if self.operator in FUNCTIONS: #operatorに関数を選んだときにノードを選択するまで再帰
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth = depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)
            
    # 突然変異
    def mutation(self):
        if random.random() < PROB_MUTATION:
            self.random_tree(grow = True, max_depth = 2)
            return
        if self.left:
            self.left.mutation()
            return
        if self.right:
            self.right.mutation()

    # 交叉 ( 部分木の入れ替え? )
    def crossover(self, other):
        if random.random() >= XO_RATE:
            return
        
        # 2nd random subtree
        second = other.scan_tree([random.randint(1, other.tree_size())], None)
        # 2nd subtree "glued" inside 1st tree
        self.scan_tree([random.randint(1, self.tree_size())], second)

    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:
                return self.build_subtree()

            self.operator  = second.operator
            self.left  = second.left
            self.right = second.right
            return None
        
        if self.left  and count[0] > 1:
            return self.left.scan_tree(count, second)
        if self.right and count[0] > 1:
            return self.right.scan_tree(count, second)
    
    def build_subtree(self):
        t = GPTree()
        t.operator = self.operator
        
        if self.left:
            t.left  = self.left.build_subtree()
        if self.right:
            t.right = self.right.build_subtree()
        return t

    def tree_size(self):
        if self.operator in TERMINALS: return 1
        
        l = self.left.tree_size()  if self.left  else 0
        r = self.right.tree_size() if self.right else 0
        return 1 + l + r

# 個体群の生成
def init_population(arg_pop_size, arg_max_depth):
    pop = []
    for md in range(3, arg_max_depth + 1):
        for i in range(int(arg_pop_size/6)): #なぜ6で割っている？
            t = GPTree() #tはインスタンス変数
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) #popに作った木を追加
        for i in range(int(arg_pop_size/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t)
    return pop

# 適応度 : 1 /(誤差の平均+1)
#   →誤差0での適応度:1、誤差:大の適応度:0。
#     「+1」は0除算エラーを避ける為
def fitness(individual, dataset):
    errors = []
    for ds in dataset:
        errors.append( abs(individual.compute_tree(ds[0]) - ds[1] ) )
        
    return 1 / ( mean(errors) +1 )

# トーナメント選択
def selection(population, fitnesses):
    tournament = []
    tournament_fitnesses = []
    # まずは、ランダムに TOURNAMENT_SIZE 個を選択
    for i in range(TOURNAMENT_SIZE):
        pop_id = random.randint(0, len(population)-1)
        tournament.append( pop_id )
        tournament_fitnesses.append( fitnesses[pop_id] )

    # その中から、適応度が最も大きいものを選択
    tmp_id = tournament_fitnesses.index( max(tournament_fitnesses) )
    return deepcopy(population[tournament[tmp_id]])

#if __name__ == '__main__':
#    main()
