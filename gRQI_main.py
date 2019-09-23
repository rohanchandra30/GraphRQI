from math import sqrt , pow
import heapq
from operator import itemgetter
import numpy as np
from numpy import *
import scipy.io
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from sklearn import svm
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import linalg as la
from math import pow
import time
from sklearn import preprocessing
import os
from collections import Counter

def generate_labels(data_path):
    all_labels = []
    sheets = ['c1.xlsx','c2.xlsx','c8.xlsx','c9.xlsx','r5.xlsx','r6.xlsx','r7.xlsx']
              # 'u8.xlsx','u9.xlsx','v1.xlsx','v2.xlsx','v8.xlsx','v9.xlsx']
    for sheet in sheets:
        labels = []
        df = pd.read_excel ( data_path+sheet)
        behavior_gt = df.as_matrix ()
        for row in range(behavior_gt.shape[0]):
            dict_item = {}
            if not isnan(behavior_gt[row,3]):
                num_traits = int(behavior_gt[row,3])
            else:
                continue
            traits = [int(behavior_gt[row][4+i]) for i in range(num_traits)]
            if (104 in traits or 105 in traits) and (106 not in traits):
                dict_item[int(behavior_gt[row][0])] = 0 # threatening
            elif 106 in traits:
                dict_item[int(behavior_gt[row][0])] = 1 # reckless
            elif 100 in traits or 101 in traits or 102 in traits or 103 in traits:
                dict_item[int(behavior_gt[row][0])] = 2 # impatient
            elif 0 in traits or 1 in traits or 2 in traits:
                dict_item[int(behavior_gt[row][0])] = 3 # careful
            elif 3 in traits or 4 in traits :
                dict_item[int(behavior_gt[row][0])] = 4 # timid
            elif 5 in traits:
                dict_item[int(behavior_gt[row][0])] = 5 # cautious
            if bool(dict_item):
                labels.append(dict_item)
        all_labels.append(labels)

    # Distribution
    vals = []
    for labels in all_labels:
        for i in range(len(labels)):
            vals.append(list(labels[i].values())[0])
    from collections import Counter
    print(Counter(vals))
    return all_labels

def pad(arrays):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    max_dim = max([embedding.shape[0]for embedding in arrays])
    combined_embedding = []
    for embedding in arrays:
        new_embedding = np.zeros([max_dim, embedding.shape[1]])

        new_embedding[:embedding.shape[0],:embedding.shape[1]] = embedding
        embedding = new_embedding
        combined_embedding.append(embedding)
    combined_embedding = np.concatenate(combined_embedding, axis=1)
    return combined_embedding.T


def converged(a,b):
    return True if la.norm(a-b) <= 1e-3 else False

def form_block(A, mu_j):
    time_1bm = time.time()
    A = (1-mu_j)*A
    A = np.hstack(  ( A, np.zeros([A.shape[0],1]) )  )
    block_matrix = np.vstack(  (  A,np.zeros([1,A.shape[1]]) )  )
    # print ( "time for computing one bm op: " , time.time () - time_1bm )
    block_matrix[-1][-1] = 1-mu_j
    return block_matrix
def GraphRQI ( U_prev, Lis_for_each_video, L_index, Lambda_prev):
    # I = eye ( A.shape[ 0 ] )
    # for j in range ( k ):
    #     u = x / linalg.norm ( x )  # normalize
    #     lam = dot ( u , dot ( A , u ) )  # Rayleigh quotient
    #     #		print j,u, lam
    #     #		print A-lam*I
    #     x = linalg.solve ( A - lam * I , u )  # inverse power iteration
    # u = x / linalg.norm ( x )
    # lam = dot ( u , dot ( A , u ) )
    # k = 3
    U = []
    L = Lis_for_each_video[L_index]
    L_prev = Lis_for_each_video[ L_index - 1 ]
    Lambda_curr = np.zeros([L.shape[0],L.shape[0]])
    delta = L[0:-1,-1]
    delta = np.expand_dims(delta, axis=1)
    # Compute sigma, sigmaTranspose
    sigma = np.eye(delta.shape[0])
    for i in range(delta.shape[0]):
        sigma[i,i] = delta[i,0]

    delta = np.hstack((delta,np.zeros([delta.shape[0],1])))
    delta = np.vstack((delta, np.zeros([1,delta.shape[1]])))
    delta[-1,-1]=1

    deltaTranspose = delta.T
    times = []
    for j in range (L.shape[0]):
        if j<4:
            times.append(time.time())
        mu_j = Lambda_prev[j,j] if j != L.shape[0]-1 else Lambda_prev[j-1,j-1]
        x_old = np.random.rand(L.shape[0],1)
        x_new = x_old / la.norm ( x_old )
        x_new = SM ( delta , deltaTranspose , form_block ( DU ( sigma , U_prev , mu_j , Lambda_prev ) , mu_j )) @ x_old

        # while not converged(x_old, x_new):
        for i in range(1):
        # Perform the rqi iterations to compute u_j
            x_old = x_new/la.norm(x_new)
        # x_new = UPDATE*x_old
            x_new = SM(delta, deltaTranspose, form_block(DU(sigma, U_prev, mu_j, Lambda_prev),mu_j))@x_old

        u_j = x_new/la.norm(x_new)
        Lambda_curr[j,j] = (u_j.T@(L@u_j)).item()
        U.append(u_j)

    U = np.array(U).T
    # Lambda_curr[-1,-1] = 1

    # print ( "time for computing one rqi op: " , (times[3]-times[2])+(times[2]-times[1])+(times[1]-times[0]))
    return U, Lambda_curr


def SM( u, v,Ainv):
    time_1SM = time.time()
    term1 =  Ainv@u
    term3 = v@Ainv
    term2 = np.eye(u.shape[1]) + term3@u
    term2inv = np.array([[term3[1,1],-1*term3[0,1]],[-1*term3[1,0],term3[0,0]]])
    # print ( "time for computing one SM op: " , time.time () - time_1SM )
    return Ainv -(term1@term2inv)@term3

def DU( sigma, U_prev, mu, Lam_prev):
    time_1DU = time.time()
    # d = [Lam-mu for Lam in Lam_prev]
    # d = [d[i]-sigma[i] for i in range(len(d))]
    # D = np.diag(d)
    D = Lam_prev + np.abs(sigma) - mu*np.eye(len(sigma))
    Ut = np.transpose(U_prev)
    for i in range(Ut.shape[0]):
        Ut[i][i] = Ut[i,i]/pow(la.norm(U_prev[:,i]),2)

    # print ( "time for computing one DU op: " , time.time () - time_1DU )
    # return (U_prev@D[0:])@(Ut)
    return (U_prev@D)@(Ut)

def computeDist ( x1 , y1 , x2 , y2 ):
    return sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )


def computeKNN ( curr_dict , ID , k, dataset ):

    if dataset == 'traf':
        ID_x , ID_y = curr_dict[ ID ]
        dists = {}
        for j in list ( curr_dict.keys () ):
            if j != ID:
                dists[ j ] = computeDist ( curr_dict[ ID ][ 0 ] , curr_dict[ ID ][ 1 ] , curr_dict[ j ][ 0 ] , curr_dict[ j ][ 1 ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )
    # print(ID,'==',list(KNN_IDs.keys()))
    else:
        lis = [el[0] for el in curr_dict]
        ind = lis.index(ID)
        ID_x , ID_y = curr_dict[ ind ][1]
        dists = {}
        for j in lis:
            if j != ID:
                dists[ j ] = computeDist ( curr_dict[ ind ][1][ 0 ] , curr_dict[ ind][1][ 1 ] , curr_dict[ lis.index(j) ][1][ 0 ] ,curr_dict[ lis.index(j) ][1][ 1 ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )

    return neighbors

def extractLi(A):
    listofTRAFLis = []
    for ad in A:
        listofLis = []
        T = ad.shape[0]
        for i in range(T-2):
            a = ad[0:i+3,0:i+3]
            d = [np.sum(a[l,:]) for l in range(a.shape[0])]  # da = sum(A,2);
            D = np.diag ( d )  # Da = diag(da);
            L = D - a
            listofLis.append(L)
        listofTRAFLis.append(listofLis)
    return listofTRAFLis


def computeA(list_of_videos, labels_list, nbrs, dataset):
    # Build A matrix of 1's and 0's.
    listOfA = []
    if dataset=='traf':
        list_of_traf_videos = list_of_videos
        for i, video in enumerate(list_of_traf_videos):
            label_list = labels_list[i]
            labels = []
            for j in range ( len ( label_list ) ):
                labels.append ( list ( label_list[ j ].keys () )[ 0 ] )
            max_ID = len(labels)
            A = np.zeros ( [ max_ID , max_ID ] )
            for idx, id in enumerate ( labels ):
                if id < max_ID:
                    for frame in video:
                        if id in list ( frame.keys () ):
                            # mark starting frame (M[ID]==0 // the ID row of M will be all zeros)
                            neighbors = computeKNN ( frame , id , nbrs, dataset )
                        for neighbor in neighbors:
                            if neighbor in labels:
                                if idx < labels.index(neighbor):
                                    A[ idx ][ labels.index(neighbor) ] = 1
                                    # A[ labels.index(neighbor)][ idx] = 1


            listOfA.append(A)
    else:
        list_of_argo_videos = list_of_videos
        labels_list = np.load('data/argo/argo_labels.npy', allow_pickle=True)
        for i, video in enumerate(list_of_argo_videos):
            label_list = labels_list[i]
            labels = []
            for j in range ( len ( label_list ) ):
                labels.append (  label_list[ j ][0] )
            max_ID = len(labels)
            A = np.zeros ( [ max_ID , max_ID ] )
            for idx, id in enumerate ( labels ):
                if id < max_ID:
                    for frame in video:
                        keys = [frame[i][0] for i in range(len(frame))]
                        if id in keys:
                            # mark starting frame (M[ID]==0 // the ID row of M will be all zeros)
                            neighbors = computeKNN ( frame , id , nbrs , dataset )
                            for neighbor in neighbors:
                                # if neighbor in labels:
                                    # if idx < labels.index(neighbor):
                                A[ idx ][ labels.index(neighbor) ] = 1
    # to do
            listOfA.append(A)

    return listOfA

def first_laplacian(index):
    return True if index==0 else False

def rayleigh_quotient(u,A):
    return np.transpose(u)@(A@u)/np.dot(u,u)

# ==============================================================START==========================================================

def main():

    # for i in range(100):
    overall_time = time.time()
    # Prepare the labels for TRAF videos.
    # 0 # impatient
    # 1 # threatening
    # 2 # reckless
    # 3 # careful
    # 4 # timid
    # 5 # cautious

    dataset = 'argo'
    nbrs = 4


    if not os.path.exists('laps_and_embs/lap.npy') or not os.path.exists('laps_and_embs/argo_lap.npy'):


        # convert TRAF file into list of dicts. Indexing of the list corresponds to the frames, ...
        # and each dict consits of key:value pairs where keys refers to the IDs in the frame, ...
        # and the values for each ID is the X-Y position
        video = [ ]
        video_list_output = [ ]

        if dataset == 'traf':
            data_path = 'data/behavior_data/'
            labels_list = generate_labels ( data_path )
            np.save ( 'labels' , labels_list )
            video_list = [ 'TRAF53_1' , 'TRAF53_2' , 'TRAF53_5' , 'TRAF53_6' , 'TRAF53_7' , 'TRAF53_8' , 'TRAF53_9' , 'TRAF29' ]
            for i in range ( len ( video_list ) ):
                video_path = 'data/behavior_data_gt/' + video_list[ i ] + '_gt.txt'
                video = [ ]
                with open ( video_path ) as file:
                    lines = file.readlines ()
                    for line in lines:
                        toks = line.split ( ',' )
                        dict_item = {}
                        for i in range ( int ( toks[ 1 ] ) ):
                            dict_item[ int ( toks[ 5 * i + 6 ] ) ] = [ int ( toks[ 5 * i + 2 ] ) ,
                                                                       int ( toks[ 5 * i + 3 ] ) ]
                        video.append ( dict_item )
                video_list_output.append ( video )
                # List of adjacency matrices corresponding to each TRAF video
            Adjacency_Matrices = computeA ( video_list_output , labels_list , nbrs , dataset)
            # List of lists: Each element of the list corresponds to a list of [L_1,L_2,...,L_T] for each TRAF video
            Laplacian_Matrices = extractLi ( Adjacency_Matrices )
            np.save ( 'laps_and_embs/lap' , Laplacian_Matrices )


        elif dataset == 'argo':
            video_list = [ 'ARGO1', 'ARGO2' ]
            data_argo = np.load('data/argo/argo_data.npy',allow_pickle=True)
            labels_argo = np.load ( 'data/argo/argo_labels.npy', allow_pickle=True )

            # List of adjacency matrices corresponding to each TRAF video
            Adjacency_Matrices = computeA(data_argo, labels_argo, nbrs,dataset)
            # List of lists: Each element of the list corresponds to a list of [L_1,L_2,...,L_T] for each TRAF video
            Laplacian_Matrices = extractLi(Adjacency_Matrices)
            np.save('laps_and_embs/argo_lap',Laplacian_Matrices)




    # ===================================MAIN ALGORITHM==================================================
    if not os.path.exists('laps_and_embs/emb.npy') or not os.path.exists('laps_and_embs/argo_emb.npy'):
        # time_start_all = time.time ()

        Laplacian_Matrices = np.load('laps_and_embs/lap.npy', allow_pickle=True) if dataset=='traf' else np.load('laps_and_embs/argo_lap.npy', allow_pickle=True)
        U_Matrices = []
        from scipy import linalg as LA
        for Lis_for_each_video in Laplacian_Matrices:
            time_start_all = time.time()
            # ListofUs = []
            for L_index,L in enumerate(Lis_for_each_video):
                if first_laplacian(L_index):
                    Lambda_prev, U_prev = la.eig(L) # need top k eigenvectors
                    Lambda_prev = np.diag(np.real(Lambda_prev))
                    # Lambda_prev = Lambda_prev[0:10,0:10]
                    U_prev= np.real(U_prev)
                else:
                    U_curr, Lambda = GraphRQI(U_prev, Lis_for_each_video, L_index, Lambda_prev)
                    Lambda_prev = Lambda
                    # ListofUs.append(U_curr)
                    U_prev = U_curr[-1]
                # Daeig , Va , X = LA.svd ( L , lapack_driver='gesvd' )
            print("time for computing spectrum for one video: ", (time.time() - time_start_all))
            U_Matrices.append(U_curr[0])
        # embedding = np.hstack ( U_Matrices )
        # embedding = embedding.T
        np.save ( 'laps_and_embs/emb' , U_Matrices ) if dataset=='traf' else np.save ( 'laps_and_embs/argo_emb' , U_Matrices )





    # =========================================ML==================================================

    all_embedding = np.load ( 'laps_and_embs/emb.npy' , allow_pickle=True ) if dataset=='traf' else np.load ( 'laps_and_embs/argo_emb.npy' , allow_pickle=True )

    # embedding = pad(all_embedding)
    # sheets_label = ['c1.xlsx','c2.xlsx','c8.xlsx','r5.xlsx','r6.xlsx','r7.xlsx','u8.xlsx','u9.xlsx']
    # video_list = [ 'TRAF53_1' , 'TRAF53_2' , 'TRAF53_5' , 'TRAF53_6' , 'TRAF53_7' , 'TRAF53_8' , 'TRAF53_9' , 'TRAF29' ]
    labels_list = np.load('laps_and_embs/labels.npy', allow_pickle=True) if dataset=='traf' else np.load('data/argo/argo_labels.npy', allow_pickle=True)
    labels = []
    index = 0
    # for index in [0,1,2,3,4,5,7]:
    if dataset == 'traf':
        for j in range(len(labels_list[index])):
            labels.append(list(labels_list[index][j].values())[0])
    else:
        for j in range(len(labels_list[index])):
            labels.append(labels_list[index][j][1])
    # for i,sheet_label in enumerate(sheets_label):
    #     spath = 'data/behavior_data/'+sheet_label
    #     df = pd.read_excel ( spath )
    #     behavior_gt = df.as_matrix ()
    #     prev_labels = list ( behavior_gt[ : , 2 ] )
    #     # labels= labels + [''] * (max_ID - len(labels))
    #     [ prev_labels.append ( 0 ) for i in range ( max_ID-1 - len ( prev_labels ) ) ]
    #     if i == 0:
    #         curr_labels = np.array ( prev_labels )
    #     else:
    #         curr_labels = np.vstack((curr_labels,np.array(prev_labels)))
    # print(Counter(labels))

    # embedding = preprocessing.scale ( embedding )
    embedding = all_embedding[index]
    [ labels .append ( 0 ) for i in range ( embedding.shape[0]- len (labels  ) ) ]
    Xtrain , Xtest = train_test_split ( embedding , test_size=0.1 )
    ytrain , ytest = train_test_split ( labels , test_size=0.1 )
    # Xtrain = embedding[0:800,:]
    # ytrain = labels[:,0:800].T
    # Xtest = embedding[800:,:]
    # ytest = labels[:,800:].T

    lr = LogisticRegression(max_iter=1)
    mlp = MLP ( hidden_layer_sizes=(10,50), max_iter=4000)
    clf = svm.SVC (max_iter=100)
    #
    iters = 1
    score = 0
    for _ in range(iters):

        #
        # lr.fit(Xtrain, ytrain)
        # y_pred = lr.predict(Xtest)
        # score += lr.score(Xtest, ytest)

        mlp.fit ( Xtrain , ytrain )
        y_pred = mlp.predict ( Xtest )
        score += mlp.score ( Xtest , ytest )

        # clf.fit ( Xtrain , ytrain )
        # y_pred = clf.predict ( Xtest )
        # score += clf.score ( Xtest , ytest )
    print(time.time() - overall_time)
    print ( score/iters)
    from sklearn.metrics import multilabel_confusion_matrix
    # cm = confusion_matrix ( ytest , y_pred )
    cm = multilabel_confusion_matrix( ytest , y_pred, labels=[0,1,2,3,4,5] )
    # print ( cm )

    f = []
    e = embedding[ : , 70 ]
    e = e.tolist()
    for i , el in enumerate ( e ):
        if i < 3 or i > 14:
            f.append ( 0 )
        else:
            f.append ( e[ i ] )
    plt.plot ( e , linewidth=16 ,alpha=0.8 )
    plt.plot(range(3,14),f[3:14], c='black',linewidth=8)
    # plt.plot ( range ( 64 , 70 ) , e[ 64:70 ] , c='black' , linewidth=8 )
    gca().set_xticklabels ( [ '' ] * len(e) )
    gca().set_yticklabels ( [ '' ] * len(e))



if __name__ == "__main__":
    main()