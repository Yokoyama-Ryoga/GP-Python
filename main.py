import settings as set
import importlib


def GP(structure="linear",POP_SIZE=60,GENERATIONS=200,XO_RATE=0.8,
       PROB_MUTATION=0.2,XO_WAY=2,SE_WAY="elite"):
    set.structure = structure
    set.POP_SIZE = POP_SIZE
    set.GENERATIONS = GENERATIONS
    set.XO_RATE = XO_RATE
    set.PROB_MUTATION = PROB_MUTATION
    set.XO_WAY = XO_WAY
    set.SE_WAY = SE_WAY
    
    if structure == "tree":
        st = importlib.import_module("tiny_gp_plus")
        st.main()
    elif structure == "linear":
        st = importlib.import_module("linear_GP")
        st.main()
        
GP("tree",60,100,0.8,0.2,2,"elite")