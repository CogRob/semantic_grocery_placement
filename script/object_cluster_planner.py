import sys
sys.path.append('../data')

from pantry_database import misc 
from scipy.special import softmax
import numpy as np

FOOD_GROUP_WEIGHT =  1.5
CONTAINER_WEIGHT = 1.2
STABLE_WEIGHT = 1
PURPOSE_WEIGHT = 1.3

def food_group_score(pantry_FG, grocery_FG):
    fg_Pvector = np.array(pantry_FG.values())
    fg_Gvector = np.array(grocery_FG.values())

    fg_score = sum(fg_Pvector * fg_Gvector) 
    fg_score +=  .5*( (pantry_FG["Fruits"] * grocery_FG["Vegetables"]) + (grocery_FG["Fruits"] * pantry_FG["Vegetables"]))
    fg_score += .5* ( (grocery_FG["Seeds_Nuts"] * pantry_FG["Butters"]) + ( pantry_FG["Seeds_Nuts"]  *  grocery_FG["Butters"]) )
    fg_score += .5* ( (pantry_FG["Roots_Tubers_Plantains"] * grocery_FG["Vegetables"]) + (grocery_FG["Roots_Tubers_Plantains"] * pantry_FG["Vegetables"]))
    return fg_score

def continer_group_score(pantry_CG, grocery_CG):
    cg_Pvector = np.array(pantry_CG.values())
    cg_Gvector = np.array(grocery_CG.values())

    cg_score = sum(cg_Pvector * cg_Gvector) 
    cg_score +=  .3*( (pantry_CG["Can"] * grocery_CG["Cylinder"]) + (grocery_CG["Can"] * pantry_CG["Cylinder"]) )
    cg_score +=  .3*( (pantry_CG["Jar"] * grocery_CG["Cylinder"]) + (grocery_CG["Jar"] * pantry_CG["Cylinder"]) )
    cg_score +=  .3*( (pantry_CG["Can"] * grocery_CG["Jar"]) + (grocery_CG["Can"] * pantry_CG["Jar"]) )
    return cg_score




pantry = [ "peanuts","cookies", "grape-jam", "spam", "apple", "onion","sugar", "bread", "ketchup"]
grocerys = ["tomato-soup", "hot-sauce", "pancake-mix", "peanut-butter"]
#cluster_list = cluster(pantry, grocerys)
#print(cluster_list)

#10,17,31, 14,32,41, 13, 36,25
# 16 , 26, 42, 38

class Grocery_cluster:
    def __init__(self,shelf1 =None,shelf2 = None, pantrys= None, grocerys=None):
        if shelf1 != None:
            self.shelf1 = shelf1 # list [ p1,p3,p9...]
            self.shelf2 = shelf2      
            self.pantry_items = pantrys 
            self.grocery_items = grocerys
        else:
            self.shelf1 =  [ "cookies", "spam", "apple","sugar", "bread"]
            self.shelf2 = [ "peanuts", "grape-jam", "onion","bread"]      
            self.pantry_items = [ "peanuts","cookies", "grape-jam", "spam", "apple", "onion","sugar", "bread", "ketchup"]
            self.grocery_items = ["tomato-soup", "hot-sauce", "pancake-mix", "peanut-butter"]
        self.clustered = None

    #def cluster(pantry_items:'list',grocery_items:'list')-> 'dict':
    def cluster(self):
        """
        Return: 
            { grocy_item: [(pantry_item, score), (pantry_item,score)], grocery....}
        """
        clusters = {}
        for grocery in self.grocery_items:
            score_list = []
            for pantry_obj in self.pantry_items:

                fg_score = FOOD_GROUP_WEIGHT * food_group_score(misc[pantry_obj]["Food_Groups"], misc[grocery]["Food_Groups"])
                continer_score = CONTAINER_WEIGHT * continer_group_score(misc[pantry_obj]["Continer"], misc[grocery]["Continer"])

                stability_score = STABLE_WEIGHT * (sum( np.array(misc[pantry_obj]["Stability"].values()) * np.array(misc[grocery]["Stability"].values()) ))
                purpose_score = PURPOSE_WEIGHT * (sum(  np.array(misc[pantry_obj]["Purpose"].values()) * np.array(misc[grocery]["Purpose"].values()) ))

                score = fg_score + continer_score + stability_score + purpose_score
                if pantry_obj in self.shelf1:
                    score_list.append(( pantry_obj, score, self.shelf1))
                elif pantry_obj in self.shelf2:
                    score_list.append(( pantry_obj, score, self.shelf2))
                    
            
            score_list.sort( key = lambda x: x[1], reverse=True)
            clusters[grocery] = score_list

        self.clustered = clusters      
        return clusters
    
  


