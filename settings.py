structure = "linear" #tree or linear

POP_SIZE        = 60    # 1世代における個体数
GENERATIONS     = 200   # 最大世代数
XO_RATE         = 0.8    # 交叉率 
PROB_MUTATION   = 0.2    # 突然変異率

#線形構造のみ設定
ADD_CHILD       = POP_SIZE / 2    # 交叉で追加する子供の数
XO_WAY          = 2      # 交叉方法 1or2点交叉
SE_WAY          = 1      # 選択方法 1:エリート選択, 2:ルーレット選択

def target_func(x):      # 目的関数
    return  x ** 3 + 2 * x ** 2 + 3 * x + 4
