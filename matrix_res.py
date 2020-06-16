# -*- coding: latin-1 -*-
import numpy as np
import numba
import time

@numba.njit
def calc(A):
    '''
    Calcul en scannant toutes les cases une par une par exemple une matrice M*M:
    1) On se place en (i,j)=(0,0)
    2) On somme la distance entre la coordonnée de 1)  et
    (i,j)=(0,0),(i,j)=(0,1),(i,j)=(0,2), ... (i,j)=(1,0),(i,j)=(1,1), ...(i,j)=(N,N) 
    N le nombre de lignes et de colonnes pour une matrice carré
    3) On se place en (i,j)=(0,1)
    4) On somme la distance entre la coordonnée de 3)  et
    (i,j)=(0,0),(i,j)=(0,1),(i,j)=(0,2), ... (i,j)=(1,0),(i,j)=(1,1), ...(i,j)=(N,N) 
    N le nombre de lignes et de colonnes pour une matrice carré
    
    Et ainsi de suite jusqu'à sommer toutes les distances.
    
    Problème de complexité : N^4 avec N nombre de lignes et colonne de la matrice
    '''
    X_AXIS=np.shape(A)[0]
    Y_AXIS=np.shape(A)[1]
    b=0
    for i in np.arange(X_AXIS):
        for j in np.arange(Y_AXIS):
            for k in np.arange(X_AXIS):
                for l in np.arange(Y_AXIS):
                    if A[i][j]==0 or A[k][l]==0:
                        b
                    elif np.sqrt((i-k)**2+(j-l)**2)!=0:
                        b=b+np.sqrt((i-k)**2+(j-l)**2)
    return b

@numba.njit
def calc2(A):
    
    '''
    Calcul en scannant toutes les cases une par une par exemple une matrice M*M:
    
    On pose c=0
    
    1) On se place en (i,j)=(0,0)
    2) On somme (dans b) la distance entre la coordonnée de 1)  et
    (i,j)=(0,0),(i,j)=(0,1),(i,j)=(0,2), ... (i,j)=(1,0),(i,j)=(1,1), ...(i,j)=(N,N) 
    N le nombre de lignes et de colonnes pour une matrice carré
    On fait alors c=c+2*b
    
    3) On se place en (i,j)=(0,1)
    4) On pose j2=j (ici j2=1), on somme (dans b) la distance entre la coordonnée de 3) et 
    (i,j)=(0,1),(i,j)=(0,2),(i,j)=(0,3), ... (i,j)=(0,N). 
    Une fois arrivé à N on pose j2=0 (retour à la ligne)
    (i,j)=(1,0),(i,j)=(1,1),(i,j)=(1,2), ... (i,j)=(1,N).
    etc jusqu'à (i,j)=(N,N)
    En fait on a juste passé la valeur (i,j)=(0,0).
    On fait alors c=c+2*b
    
    5) On se place en (i,j)=(0,2)
    6) On pose j2=j (ici j2=2), on somme (dans b) la distance entre la coordonnée de 3) et 
    (i,j)=(0,2),(i,j)=(0,3),(i,j)=(0,4), ... (i,j)=(0,N). 
    Une fois arrivé à N on pose j2=0 (retour à la ligne)
    (i,j)=(1,0),(i,j)=(1,1),(i,j)=(1,2), ... (i,j)=(1,N).
    etc jusqu'à (i,j)=(N,N)
    En fait on a juste passé les valeurs (i,j)=(0,0) et (i,j)=(0,1).
    On fait alors c=c+2*b
    
    
    Et ainsi de suite jusqu'à sommer toutes les distances.
    Le fait de faire cette algorithme et de faire 2*b permet d'éviter la moitié des calculs par symétrie. 
    
    Problème de complexité : (N/2)^4 avec N nombre de lignes et colonne de la matrice
    On divise par 2 le temps de calcul théoriquement.
    '''
    
    b=0
    c=0
    X_AXIS=np.shape(A)[0]
    Y_AXIS=np.shape(A)[1]
    
    
    for i in np.arange(X_AXIS):
        
        for j in np.arange(Y_AXIS):
            j2=j
            if A[i][j]==0 :
                pass
            else:           
                for k in np.arange(i,X_AXIS):

                    for l in np.arange(j2,Y_AXIS):
                        if A[k][l]==0:
                            pass
                        else:
                            b=b+np.sqrt((i-k)**2+(j-l)**2)


                    j2=0
                    c=c+2*b
                    b=0
    return c


@numba.njit
def calc3(A):
    '''
    On calcule préalablement la matrice de distance : voir ligne au dessus 
    Pour N=3, la matrice de distance B est :
    
    B=([[0.        , 1.        , 2.        ],
       [1.        , 1.41421356, 2.23606798],
       [2.        , 2.23606798, 2.82842712]])
       
    On va ensuite chercher dans cette matrice les distances que l'on souhaite avec la formule suivante : 
    B[abs(i-k)][abs(j-l)] (Formule compliqué à trouver.)
    Cela évite de calculer la racine carré ainsi que les carrés à chaque fois mais juste la somme.
    
    (i,j) : première coordonnée
    (k,l) : deuxième coordonnée
    
    la distance entre (0,1) et (1,2) est alors B[1][1] soit 1.414 soit sqrt(2) et on peut calculer que c'est bien le cas. 
    Test sur beaucoup de valeur et cela marche. 
    
    On somme sur tout les (i,j) et tout les (k,l) et ça marche.
    '''
    N=len(A)
    x=np.reshape(np.repeat(np.arange(0,N),N),(N,N))
    x2=np.transpose(x)
    B=np.sqrt(np.multiply(x,x)+np.multiply(x2,x2))

    c=0
    X_AXIS=np.shape(A)[0]
    Y_AXIS=np.shape(A)[1]
    
    
    
    for i in np.arange(X_AXIS):
        
        for j in np.arange(Y_AXIS):
            
            for k in np.arange(X_AXIS):
                
                for l in np.arange(Y_AXIS):
                    if A[i][j]==0 or A[k][l]==0:
                        pass
                    else:
                        c=c+B[abs(i-k)][abs(j-l)]
                    
                    
                        
    return c

@numba.njit
def calc4(A):
    '''
    On calcule préalablement la matrice de distance : voir ligne au dessus 
    Pour N=3, la matrice de distance B est :
    
    B=([[0.        , 1.        , 2.        ],
       [1.        , 1.41421356, 2.23606798],
       [2.        , 2.23606798, 2.82842712]])
       
    La méthode ici est semblable à la fonction calc3() cependant au lieu de sommer chaque élément
    on va creer une matrice B2, et on va la remplir en fonction des distances de la matrice B et en fonction des coordonnées.
    Quand on l'aura fait pour tout les (j,k), on somme tout les éléments de cette matrice. et on passe à un autre (i,j).
    
    Pour une matrice 3x3 et pour (i,j)= ( 0 , 0 )
    B1=  [[0.         1.         2.        ]
          [1.         1.41421356 2.23606798]
          [2.         2.23606798 2.82842712]]
          
    Pour une matrice 3x3 et pour (i,j)= ( 0 , 1 )
    B1=  [[1.         0.         1.        ]
          [1.41421356 1.         1.41421356]
          [2.23606798 2.         2.23606798]]
          
    Pour une matrice 3x3 et pour (i,j)= ( 1 , 2 ) 
    B1=  [[2.23606798 1.41421356 1.        ]
          [2.         1.         0.        ]
          [2.23606798 1.41421356 1.        ]]
    '''
    N=len(A)
    x=np.reshape(np.repeat(np.arange(0,N),N),(N,N))
    x2=np.transpose(x)
    B=np.sqrt(np.multiply(x,x)+np.multiply(x2,x2))
    B1=np.zeros(np.shape(A))
    c=0
    X_AXIS=np.shape(A)[0]
    Y_AXIS=np.shape(A)[1]
    
    
    
    for i in np.arange(X_AXIS):
        #print(i/X_AXIS*100)
        #print('%')
        
        for j in np.arange(Y_AXIS):
            if A[i][j]==0:
                pass
            else:
                for k in np.arange(X_AXIS):

                    for l in np.arange(Y_AXIS):
                        if A[i][j]==0:
                            break
                        else:
                            B1[k][l]=B[abs(i-k)][abs(j-l)]
                c=c+np.sum(np.multiply(B1,A))
                        
    return c

# Pour une matrice 20 * 20
N=20
calc_time=[]
calc2_time=[]
calc3_time=[]
calc4_time=[]
size_matrix=[]
A=np.ones((N,N))
sum_calc=calc(A)
sum_calc2=calc2(A)
sum_calc3=calc3(A)
sum_calc4=calc4(A)
for N in range(0,450,50):
    size_matrix.append(N)

    A=np.ones((N,N))

    start=time.time()
    sum_calc=calc(A)
    stop=time.time()
    calc_time.append(stop-start)
    print("calc() : Temps écoulé : ", stop-start," secondes", "pour une matrice ",N,"x",N)
    print("Voici la valeur de la somme",sum_calc)

    start=time.time()
    sum_calc2=calc2(A)
    stop=time.time()
    calc2_time.append(stop-start)
    print("calc2() : Temps écoulé : ", stop-start," secondes", "pour une matrice ",N,"x",N)
    print("Voici la valeur de la somme",sum_calc2)

    start=time.time()
    sum_calc3=calc3(A)
    stop=time.time()
    calc3_time.append(stop-start)
    print("calc3() : Temps écoulé : ", stop-start," secondes", "pour une matrice ",N,"x",N)
    print("Voici la valeur de la somme",sum_calc3)

    start=time.time()
    sum_calc4=calc4(A)
    stop=time.time()
    calc4_time.append(stop-start)
    print("calc4() : Temps écoulé : ", stop-start," secondes", "pour une matrice ",N,"x",N)
    print("Voici la valeur de la somme",sum_calc4)
    print(" ", end ="\n") 

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(size_matrix,calc_time,'o',label='Calcul direct')
plt.plot(size_matrix,calc2_time,'x',label='Calcul symétrique')
plt.plot(size_matrix,calc3_time,'*',label='Calcul matriciel')
plt.plot(size_matrix,calc4_time,'+',label='Calcul matriciel Bis')
plt.title("Execution avec Numba (Compilateur JIT) pour une matrice NxN",fontsize=14)
plt.xlabel('N',fontsize=12)
plt.ylabel('Temps (s)',fontsize=12)
plt.legend(fontsize=12)
plt.show()