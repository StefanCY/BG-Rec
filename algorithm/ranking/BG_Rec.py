#coding=utf-8
from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import shuffle, choice, random
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine
from math import log, exp
import gensim.models.word2vec as w2v



class BG_Rec(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(BG_Rec, self).readConfiguration()
        options = config.LineConfig(self.config['BG_Rec'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.alpha = float(options['-a'])
        self.epoch = int(options['-ep'])
        self.neg = int(options['-neg'])
        self.rate = float(options['-r'])

    def printAlgorConfig(self):
        super(BG_Rec, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def readNegativeFeedbacks(self):
        self.negative = defaultdict(list) # negative feedback user:[item,item,...]
        self.nItems = defaultdict(list)   # negative feedback item:[user,user,...]
        self.Hyper_PositiveSet = defaultdict(dict) #Hyper_PositiveSet、Hyper_NegSets中的rating没有经过二值化 (userName,itemName,rating)
        self.Hyper_NegSets = defaultdict(dict)
        self.p_UI_Net = defaultdict(list) #p_UI_Net[user1]表示在positive数据集中与user1有记录的item列表
        self.p_IU_Net = defaultdict(list) #p_IU_Net[item1]表示在positive数据集中与item1有记录的user列表
        self.p_user = {}
        self.p_id2user = {}
        self.p_item = {}
        self.p_id2item = {}
        #negtive
        self.n_UI_Net = defaultdict(list)
        self.n_IU_Net = defaultdict(list)
        self.n_user = {}
        self.n_id2user = {}
        self.n_item = {}
        self.n_id2item = {}

        filename = self.config['ratings'][:-6]+'_n.txt'
        with open(filename) as f:
            for line in f:
                items = line.strip().split()
                self.Hyper_NegSets[items[0]][items[1]] = items[2]
                self.n_UI_Net[items[0]].append(items[1])
                self.n_IU_Net[items[1]].append(items[0])
                if items[0] not in self.n_user:
                    self.n_user[items[0]] = len(self.n_user)
                    self.n_id2user[self.n_user[items[0]]] = items[0]
                if items[1] not in self.n_item:
                    self.n_item[items[1]] = len(self.n_item)
                    self.n_id2item[self.n_item[items[1]]] = items[1]

                self.negative[items[0]].append(items[1])
                self.nItems[items[1]].append(items[0])
                if items[0] not in self.data.user:
                    self.data.user[items[0]]=len(self.data.user)
                    self.data.id2user[self.data.user[items[0]]] = items[0]

        filename_hyperPositive = self.config['ratings'][:-6]+'_h_p.txt'
        with open(filename_hyperPositive) as f:
            for line in f:
                items = line.strip().split()
                self.Hyper_PositiveSet[items[0]][items[1]] = items[2]
                self.p_UI_Net[items[0]].append(items[1])
                self.p_IU_Net[items[1]].append(items[0])
                if items[0] not in self.p_user:
                    self.p_user[items[0]] = len(self.p_user)
                    self.p_id2user[self.p_user[items[0]]] = items[0]
                if items[1] not in self.p_item:
                    self.p_item[items[1]] = len(self.p_item)
                    self.p_id2item[self.p_item[items[1]]] = items[1]

    def initModel(self):
        super(BG_Rec, self).initModel()
        self.positive = defaultdict(list) # positive feedback user:[item,item,...]
        self.pItems = defaultdict(list)   # positive feedback item:[user,user,...]
        for user in self.data.trainSet_u:
            for item in self.data.trainSet_u[user]:
                self.positive[user].append(item)
                self.pItems[item].append(user)

        self.readNegativeFeedbacks()

        self.P = np.ones((len(self.data.user), self.embed_size))*0.1  # latent user matrix
        self.threshold = {}
        self.avg_sim = {}
        self.thres_d = dict.fromkeys(self.data.user.keys(),0) #derivatives for learning thresholds
        self.thres_count = dict.fromkeys(self.data.user.keys(),0)

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict) # positive record set (user, item, 1)
        self.NegSets = defaultdict(dict)     # negative record set (user, item, 1)

        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                self.PositiveSet[user][item] = 1

        for user in self.data.user:
            for item in self.negative[user]:
                if self.data.item.has_key(item):
                    self.NegSets[user][item] = 1

        for user in self.negative:
            if user not in self.positive:
                self.positive[user] = []

    def hyperWalks(self):
        #self.walkCount = 100
        #self.walkLength = 40
        self.walkCount = 100
        self.walkLength = 40
        self.Hyper_alpha = 0.9
        self.UFNet = defaultdict(list) # a -> b #a trusts b
        for u in self.social.followees:
            s1 = set(self.social.followees[u])
            for v in self.social.followees[u]:
                if v in self.social.followees:  # make sure that v has out links
                    if u <> v:
                        s2 = set(self.social.followees[v])
                        weight = len(s1.intersection(s2))
                        self.UFNet[u] += [v] * (weight + 1)
        #positive random walk
        print 'Generating random walks on hypergraph... (Positive)'
        self.W_p = np.zeros((len(self.p_UI_Net),len(self.p_IU_Net))) #|V|x|E|
        self.R_p = np.zeros((len(self.p_IU_Net),len(self.p_UI_Net))) #|E|x|V|
        print 'len(self.p_UI_Net)',len(self.p_UI_Net)
        print 'len(self.p_IU_Net)',len(self.p_IU_Net)
        self.v_weight_p = {}
        self.v_weight_n = {}
        #经过v的所有e的权重之和，e的权重为e包含的v的个数
        for v in self.n_UI_Net:
            w = 0
            for e in self.n_UI_Net[v]:
                w += 1.0/len(self.n_IU_Net[e])
            self.v_weight_n[v] = float(w)

        for v in self.p_UI_Net:
            w = 0
            for e in self.p_UI_Net[v]:
                w += 1.0/len(self.p_IU_Net[e])
            self.v_weight_p[v] = float(w)
        #W_p(i,j)
        for v in self.p_UI_Net:
            for e in self.p_UI_Net[v]:
                i = self.p_user[v]
                j = self.p_item[e]
                self.W_p[i][j] = float(1.0/len(self.p_IU_Net[e]))/self.v_weight_p[v]

        #R_p(j,i)
        for e in self.p_IU_Net:
            delta = 0
            for x in self.p_IU_Net[e]:
                delta += float(self.Hyper_PositiveSet[x][e])
            for v in self.p_IU_Net[e]:
                i = self.p_user[v]
                j = self.p_item[e]
                self.R_p[j][i] = float(self.Hyper_PositiveSet[v][e])/delta
        #matrix P shows the probability of one node jump to another node, P(i,j) is the the probability of node i jump to node j.
        self.P_p = np.matmul(self.W_p,self.R_p) #|V|x|V|

        self.p_v2v = defaultdict(list)
        for v1 in self.p_user:
            for v2 in self.p_user:
                if v1 <> v2:
                    i = self.p_user[v1]
                    j = self.p_user[v2]
                    num = int(self.P_p[i][j]*1000)
                    self.p_v2v[v1] += [v2]*num
        
        self.pWalks = []
        for user in self.p_user:
            for i in range(self.walkCount):
                path = [user]
                nextNode = None
                for j in range(self.walkLength):
                    curentNode = path[-1]
                    if random() < self.Hyper_alpha:
                        if len(self.p_v2v[curentNode]) >0:
                            nextNode = choice(self.p_v2v[curentNode])
                            path.append(nextNode)
                        elif curentNode in self.UFNet:
                            nextNode = choice(self.UFNet[curentNode])
                            path.append(nextNode)
                    else:
                        if curentNode in self.UFNet:
                            nextNode = choice(self.UFNet[curentNode])
                            path.append(nextNode)
                        elif len(self.p_v2v[curentNode]) >0:
                            nextNode = choice(self.p_v2v[curentNode])
                            path.append(nextNode)
                self.pWalks.append(path)
        shuffle(self.pWalks)

        #negative random walk
        print 'Generating random walks on hypergraph... (Negative)'
        self.W_n = np.zeros((len(self.n_UI_Net),len(self.n_IU_Net))) #|V|x|E|
        self.R_n = np.zeros((len(self.n_IU_Net),len(self.n_UI_Net))) #|E|x|V|

        #W_p(i,j)
        for v in self.n_UI_Net:
            for e in self.n_UI_Net[v]:
                i = self.n_user[v]
                j = self.n_item[e]
                self.W_n[i][j] = float(1.0/len(self.n_IU_Net[e]))/self.v_weight_n[v]
        #R_p(j,i)
        for e in self.n_IU_Net:
            delta = 0
            for x in self.n_IU_Net[e]:
                delta += float(self.Hyper_NegSets[x][e])
            for v in self.n_IU_Net[e]:
                i = self.n_user[v]
                j = self.n_item[e]
                self.R_n[j][i] = float(self.Hyper_NegSets[v][e])/delta
        #matrix P shows the probability of one node jump to another node, P(i,j) is the the probability of node i jump to node j.
        self.P_n = np.matmul(self.W_n,self.R_n) #|V|x|V|
        self.n_v2v = defaultdict(list)
        for v1 in self.n_user:
            for v2 in self.n_user:
                if v1 <> v2:
                    i = self.n_user[v1]
                    j = self.n_user[v2]
                    num = int(self.P_n[i][j]*1000)
                    self.n_v2v[v1] += [v2]*num
        self.nWalks = []
        for user in self.n_user:
            for i in range(self.walkCount):
                path = [user]
                nextNode = None
                for j in range(self.walkLength):
                    curentNode = path[-1]
                    if random() < self.Hyper_alpha:
                        if len(self.n_v2v[curentNode]) >0:
                            nextNode = choice(self.n_v2v[curentNode])
                            path.append(nextNode)
                        elif curentNode in self.UFNet:
                            nextNode = choice(self.UFNet[curentNode])
                            path.append(nextNode)
                    else:
                        if curentNode in self.UFNet:
                            nextNode = choice(self.UFNet[curentNode])
                            path.append(nextNode)
                        elif len(self.n_v2v[curentNode]) >0:
                            nextNode = choice(self.n_v2v[curentNode])
                            path.append(nextNode)
                self.nWalks.append(path)
        shuffle(self.nWalks)
        print 'pwalks:', len(self.pWalks)
        print 'nwalks:', len(self.nWalks)

    def computeSimilarity(self):
        # Training get top-k friends
        self.G = np.random.rand(self.data.trainingSize()[0], self.walkDim) * 0.1
        self.W = np.random.rand(self.data.trainingSize()[0], self.walkDim) * 0.1
        print 'Generating user embedding...'
        self.pTopKSim = {}
        self.nTopKSim = {}
        self.pSimilarity = defaultdict(dict)
        self.nSimilarity = defaultdict(dict)
        pos_model = w2v.Word2Vec(self.pWalks, size=self.walkDim, window=5, min_count=0, iter=10)
        neg_model = w2v.Word2Vec(self.nWalks, size=self.walkDim, window=5, min_count=0, iter=10)

        for user in self.positive:
            uid = self.data.user[user]
            try:
                self.W[uid] = pos_model.wv[user]  # get positive user embedding from Word2Vec
            except KeyError:
                continue
        for user in self.negative:
            uid = self.data.user[user]
            try:
                self.G[uid] = neg_model.wv[user]  # get negative user embedding from Word2Vec
            except KeyError:
                continue
        print 'User embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0
        for user1 in self.positive:
            uSim = []
            i += 1
            if i % 200 == 0:
                print i, '/', len(self.positive)
            vec1 = self.W[self.data.user[user1]]
            for user2 in self.positive:
                if user1 <> user2:
                    vec2 = self.W[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            self.threshold[user1] = fList[self.topK / 2][1]
            for pair in fList:
                self.pSimilarity[user1][pair[0]] = pair[1]
            self.pTopKSim[user1] = [item[0] for item in fList]
            self.avg_sim[user1] = sum([item[1] for item in fList][:self.topK / 2]) / (self.topK / 2)

        i = 0
        for user1 in self.negative:
            uSim = []
            i += 1
            if i % 200 == 0:
                print i, '/', len(self.negative)
            vec1 = self.G[self.data.user[user1]]
            for user2 in self.negative:
                if user1 <> user2:
                    vec2 = self.G[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            for pair in fList:
                self.nSimilarity[user1][pair[0]] = pair[1]
            self.nTopKSim[user1] = [item[0] for item in fList]

        self.trueTopKFriends = defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.trueTopKFriends[user] = trueFriends
            self.pTopKSim[user] = list(set(self.pTopKSim[user]).difference(set(trueFriends)))

    def updateSets(self):
        self.JointSet = defaultdict(dict)
        self.PS_Set = defaultdict(dict)
        for user in self.data.user:
            if user in self.trueTopKFriends:
                for friend in self.trueTopKFriends[user]:
                    if friend in self.data.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.NegSets[user]:
                                self.JointSet[user][item] = friend

            if self.pTopKSim.has_key(user):
                for friend in self.pTopKSim[user][:self.topK]:
                    if friend in self.data.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                    and item not in self.NegSets[user]:
                                self.PS_Set[user][item] = friend

            if self.nTopKSim.has_key(user):
                for friend in self.nTopKSim[user][:self.topK]:
                    if friend in self.data.user and self.nSimilarity[user][friend]>=self.threshold[user]:
                        for item in self.negative[friend]:
                            if item in self.data.item:
                                if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                        and item not in self.PS_Set[user]:
                                    self.NegSets[user][item] = friend

    def buildModel(self):
        self.hyperWalks()
        self.computeSimilarity()

        print 'Decomposing...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            self.updateSets()
            itemList = self.data.item.keys()
            for user in self.PositiveSet:
                #itemList = self.NegSets[user].keys()
                kItems = self.JointSet[user].keys()
                pItems = self.PS_Set[user].keys()
                nItems = self.NegSets[user].keys()

                u = self.data.user[user]

                for item in self.PositiveSet[user]:
                    i = self.data.item[item]
                    selectedItems = [i]
                    #select items from different sets
                    if len(kItems) > 0:
                        item_k = choice(kItems)                        
                        uf = self.JointSet[user][item_k]
                        k = self.data.item[item_k]
                        selectedItems.append(k)
                        self.optimization_thres(u,i,k,user,uf)
                    if len(pItems)>0:
                        item_p = choice(pItems)
                        p = self.data.item[item_p]
                        selectedItems.append(p)

                    item_r = choice(itemList)
                    while item_r in self.PositiveSet[user] or item_r in self.JointSet[user]\
                        or item_r in self.PS_Set[user] or item_r in self.NegSets[user]:
                        item_r = choice(itemList)
                    r = self.data.item[item_r]
                    selectedItems.append(r)

                    if len(nItems)>0:
                        item_n = choice(nItems)
                        n = self.data.item[item_n]
                        selectedItems.append(n)

                    #optimization
                    for ind,item in enumerate(selectedItems[:-1]):
                        self.optimization(u,item,selectedItems[ind+1])


                if self.thres_count[user]>0:
                    self.threshold[user] -= self.lRate * self.thres_d[user] / self.thres_count[user]
                    self.thres_d[user]=0
                    self.thres_count[user]=0
                    li = [sim for sim in self.pSimilarity[user].values() if sim>=self.threshold[user]]
                    if len(li)==0:
                        self.avg_sim[user] = self.threshold[user]
                    else:
                        self.avg_sim[user]= sum(li)/(len(li)+0.0)

                for friend in self.trueTopKFriends[user]:
                    if self.pSimilarity[user][friend]>self.threshold[user]:
                        u = self.data.user[user]
                        f = self.data.user[friend]
                        self.P[u] -= self.alpha*self.lRate*(self.P[u]-self.P[f])

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                 break
            print self.foldInfo,'iteration:',iteration
        self.ranking_performance()


    def optimization(self, u, i, j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]

    def optimization_thres(self, u, i, j,user,friend):
        try:
            g_theta = sigmoid((self.pSimilarity[user][friend]-self.threshold[user])/(self.avg_sim[user]-self.threshold[user]))
        except OverflowError:
            print 'threshold',self.threshold[user],'smilarity',self.pSimilarity[user][friend],'avg',self.avg_sim[user]
            print (self.pSimilarity[user][friend]-self.threshold[user]),(self.avg_sim[user]-self.threshold[user])
            print (self.pSimilarity[user][friend]-self.threshold[user])/(self.avg_sim[user]-self.threshold[user])
            exit(-1)

        s = sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))/(1+g_theta))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        t_derivative = -g_theta*(1-g_theta)*(1-s)*(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))\
                       *(self.pSimilarity[user][friend]-self.avg_sim[user])/(self.avg_sim[user]-
                       self.threshold[user])**2/(1+g_theta)**2 + 0.005*self.threshold[user]
        #print 'derivative', t_derivative
        self.thres_d[user] += t_derivative
        self.thres_count[user] += 1

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items