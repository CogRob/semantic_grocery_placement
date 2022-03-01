import sys
import math
sys.path.append('../data')

from pantry_database import misc 
from scipy.special import softmax
from scipy import spatial
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader

FOOD_GROUP_WEIGHT =  1.5
CONTAINER_WEIGHT = 1.2
STABLE_WEIGHT = 1
PURPOSE_WEIGHT = 1.3
WORD2VECT_WEIGHT = 1.5

def food_group_score(pantry_FG, grocery_FG):
    fg_Pvector =  np.array(pantry_FG.values()).astype(np.float)
    fg_Gvector =  np.array(grocery_FG.values()).astype(np.float)

    #fg_score = sum(fg_Pvector * fg_Gvector) 
    fg_score = 1- spatial.distance.cosine(fg_Pvector, fg_Gvector)
    # fg_score +=  .5*( (pantry_FG["Fruits"] * grocery_FG["Vegetables"]) + (grocery_FG["Fruits"] * pantry_FG["Vegetables"]))
    # fg_score += .5* ( (grocery_FG["Seeds_Nuts"] * pantry_FG["Butters"]) + ( pantry_FG["Seeds_Nuts"]  *  grocery_FG["Butters"]) )
    # fg_score += .5* ( (pantry_FG["Roots_Tubers_Plantains"] * grocery_FG["Vegetables"]) + (grocery_FG["Roots_Tubers_Plantains"] * pantry_FG["Vegetables"]))
    return fg_score

def continer_group_score(pantry_CG, grocery_CG):
    cg_Pvector = np.array(pantry_CG.values())
    cg_Gvector = np.array(grocery_CG.values())

    #cg_score = sum(cg_Pvector * cg_Gvector)
    cg_score = 1- spatial.distance.cosine(cg_Pvector, cg_Gvector) 
    # cg_score +=  .3*( (pantry_CG["Can"] * grocery_CG["Cylinder"]) + (grocery_CG["Can"] * pantry_CG["Cylinder"]) )
    # cg_score +=  .3*( (pantry_CG["Jar"] * grocery_CG["Cylinder"]) + (grocery_CG["Jar"] * pantry_CG["Cylinder"]) )
    # cg_score +=  .3*( (pantry_CG["Can"] * grocery_CG["Jar"]) + (grocery_CG["Can"] * pantry_CG["Jar"]) )
    return cg_score

def word2vec_score(glove_wiki_vec,pantry,grocery):
    complex_words = [ "gummy-vitamins", "baby-formula", "condensed-milk", "alfredo-sauce","olive-oil", "granola-bars","pancake-mix", "peanut-butter", "grape-jam","bread-crumbs", "hot-sauce", "tomato-soup", "muffin-mix"]
    wd_score = 0 

    if pantry not in complex_words and grocery not in complex_words:
        wd_score += glove_wiki_vec.similarity(pantry, grocery)
    else:
        if pantry in complex_words and grocery in complex_words:
            pantry = pantry.split("-")
            grocery = grocery.split("-")
            pantry = glove_wiki_vec[pantry[0]] +  glove_wiki_vec[pantry[1]]
            grocery = glove_wiki_vec[grocery[0]] +  glove_wiki_vec[grocery[1]]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
        elif pantry in complex_words:
            pantry = pantry.split("-")
            pantry = glove_wiki_vec[pantry[0]] +  glove_wiki_vec[pantry[1]]
            grocery = glove_wiki_vec[grocery]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
        elif grocery in complex_words:
            grocery = grocery.split("-")
            grocery = glove_wiki_vec[grocery[0]] +  glove_wiki_vec[grocery[1]]
            pantry = glove_wiki_vec[pantry]
            pantry = np.array(pantry)
            grocery = np.array(grocery)
            wd_score += pantry.dot(grocery)/np.linalg.norm(grocery)/np.linalg.norm(pantry)
    
    return wd_score

# #def cluster(pantry_items:'list',grocery_items:'list')-> 'dict':
def cluster(pantry_items, grocery_items):
    """
    Return: 
        { grocy_item: [(pantry_item, score), (pantry_item,score)], grocery....}
    """
    glove_wiki_vec = gensim.downloader.load('glove-wiki-gigaword-300')
    clusters = {}
    for grocery in grocery_items:
        score_list = []
        for pantry_obj in pantry_items:

            fg_score = FOOD_GROUP_WEIGHT * food_group_score(misc[pantry_obj]["Food_Groups"], misc[grocery]["Food_Groups"])
            continer_score = CONTAINER_WEIGHT * continer_group_score(misc[pantry_obj]["Continer"], misc[grocery]["Continer"])

            stability_score = STABLE_WEIGHT * (sum(np.array(misc[pantry_obj]["Stability"].values()) * np.array(misc[grocery]["Stability"].values()) ))
            purpose_score = PURPOSE_WEIGHT * (sum(np.array(misc[pantry_obj]["Purpose"].values()) * np.array(misc[grocery]["Purpose"].values()) ))

            wdVec_score = WORD2VECT_WEIGHT * word2vec_score(glove_wiki_vec ,pantry_obj,grocery)

            score = fg_score + continer_score + stability_score + purpose_score + wdVec_score
            score_list.append(( pantry_obj, score))
            # if pantry_obj in self.shelf1:
            #     score_list.append(( pantry_obj, score, self.shelf1))
            # elif pantry_obj in self.shelf2:
            #     score_list.append(( pantry_obj, score, self.shelf2))
                
        
        score_list.sort( key = lambda x: x[1], reverse=True)
        clusters[grocery] = score_list

    #self.clustered = clusters      
    return clusters

def cluster_covariance_matrix():
    glove_wiki_vec = gensim.downloader.load('glove-wiki-gigaword-300')
    variables_len = 5
    num_objs = len(misc)
    X = np.zeros((num_objs*num_objs, variables_len))
    for c in range(0,len(X[0]-1)):
        for r in range(0,len(X)):
            var = 0.0
            current_obj = misc.keys()[r]
            #print(misc[current_obj]["Food_Groups"].values())
            for obj in misc.keys():
                if c == 0:
                    var += food_group_score(misc[current_obj]["Food_Groups"], misc[obj]["Food_Groups"])
                if c == 1:
                    var += continer_group_score(misc[current_obj]["Continer"], misc[obj]["Continer"])
                if c == 2:
                    var +=  sum(np.array(misc[current_obj]["Stability"].values()) * np.array(misc[obj]["Stability"].values()) )
                if c == 3:
                    var += sum(np.array(misc[current_obj]["Purpose"].values()) * np.array(misc[obj]["Purpose"].values()) )

                #var = var / num_objs
                X[r][c] = var 

    complex_words = [ "gummy-vitamins", "baby-formula", "condensed-milk", "alfredo-sauce","olive-oil", "granola-bars","pancake-mix", "peanut-butter", "grape-jam","bread-crumbs", "hot-sauce", "tomato-soup", "muffin-mix"]
    for r in range(0,len(X)):
        #print(misc.keys()[r])
        wd_score = 0 
        for obj in misc.keys():
            current_obj = misc.keys()[r]
            if current_obj not in complex_words and obj not in complex_words:
                wd_score += glove_wiki_vec.similarity( current_obj, obj)
            else:
                if current_obj in complex_words and obj in complex_words:
                    current_obj = current_obj.split("-")
                    obj = obj.split("-")
                    current_obj = glove_wiki_vec[current_obj[0]] +  glove_wiki_vec[current_obj[1]]
                    obj = glove_wiki_vec[obj[0]] +  glove_wiki_vec[obj[1]]
                    current_obj = np.array(current_obj)
                    obj = np.array(obj)
                    wd_score += current_obj.dot(obj)/np.linalg.norm(obj)/np.linalg.norm(current_obj)
                elif current_obj in complex_words:
                    current_obj = current_obj.split("-")
                    current_obj = glove_wiki_vec[current_obj[0]] +  glove_wiki_vec[current_obj[1]]
                    obj = glove_wiki_vec[obj]
                    current_obj = np.array(current_obj)
                    obj = np.array(obj)
                    wd_score += current_obj.dot(obj)/np.linalg.norm(obj)/np.linalg.norm(current_obj)
                elif obj in complex_words:
                    obj = obj.split("-")
                    obj = glove_wiki_vec[obj[0]] +  glove_wiki_vec[obj[1]]
                    current_obj = glove_wiki_vec[current_obj]
                    current_obj = np.array(current_obj)
                    obj = np.array(obj)
                    wd_score += current_obj.dot(obj)/np.linalg.norm(obj)/np.linalg.norm(current_obj)
            #wd_score = wd_score/num_objs
            X[r][variables_len-1] = wd_score

    # X_mean = [ sum(X[:,0])/num_objs, sum(X[:,1])/num_objs , sum(X[:,2])/num_objs,
    #             sum(X[:,3])/num_objs, sum(X[:,4])/num_objs]
    # X_mean = np.array(X_mean)
    print("printing X")
    print(X)

    # print(X_mean)
    C = np.cov(X, rowvar=False)
    # cov_sum = np.array([0.,0.,0.,0.])
    # for i in range(0, num_objs):
    #     V = X[i,:] - X_mean
    #     V_t = np.transpose(V)
    #     cov_sum += V.dot(V_t) 
    # print(cov_sum/(num_objs-1))
    print("Pringint C")
    print(C)

pantry = [ "hot-sauce","cookies", "ketchup", "muffin-mix", "granola-bars", "beans", "pringles"]
grocerys = [ "pringles", "spam", "sugar", "mustard"]
#grocerys= ["tomato-soup", "hot-sauce", "pancake-mix", "peanut-butter"]
#cluster_list = cluster(pantry, grocerys)
#print(cluster_list)
cluster_covariance_matrix()


class Grocery_cluster:
    def __init__(self,shelf1 =None,shelf2 = None, pantrys= None, grocerys=None):
        if shelf1 != None:
            self.shelf1 = shelf1 # list ( [ p1,p3,p9...], height)
            self.shelf2 = shelf2      
            self.pantry_items = pantrys 
            self.grocery_items = grocerys
        else:
            self.shelf1 =  [ "cookies", "pringles", "granola-bars"]
            self.shelf2 = ["spam", "muffin-mix","mustard","beans"]      
            self.pantry_items = [ "peanuts","cookies", "grape-jam", "spam", "apple", "onion","sugar", "bread", "ketchup"]
            self.grocery_items = ["tomato-soup", "hot-sauce", "pancake-mix", "peanut-butter"]
        self.clustered = None
        self.glove_wiki_vec = gensim.downloader.load('glove-wiki-gigaword-300')

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
                
                wdVec_score = WORD2VECT_WEIGHT * word2vec_score(self.glove_wiki_vec, pantry_obj,grocery)

                score = fg_score + continer_score + stability_score + purpose_score + wdVec_score
                if pantry_obj in self.shelf1[0]:
                    score_list.append(( pantry_obj, score, self.shelf1[1]))
                elif pantry_obj in self.shelf2[0]:
                    score_list.append(( pantry_obj, score, self.shelf2[1]))
                    
                
            score_list.sort( key = lambda x: x[1], reverse=True)
            clusters[grocery] = score_list

        self.clustered = clusters      
        return clusters
    



