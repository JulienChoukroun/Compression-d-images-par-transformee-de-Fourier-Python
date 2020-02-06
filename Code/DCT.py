from PIL import Image
import numpy as np
import math
from scipy.fftpack import dct,idct

#calcul de la matrice D (DCT 2)
def CalculDsansP(M):
    D=np.zeros([8,8])
    for k in range(7):
        for l in range(7):
            if k==0:
                Ck=1/np.sqrt(2)
            else:
                Ck=1
            if l==0:
                Cl=1/np.sqrt(2)
            else:
                Cl=1
            for i in range(7):
                for j in range(7):
                    D[k,l]=D[k,l]+0.25*Ck*Cl*M[i,j]*np.cos(((2*i+1)*k*np.pi)/16)*np.cos(((2*j+1)*l*np.pi)/16)
    return D


image=Image.open("test.jpg")
#image.show()
longueur=image.size[0]
largeur=image.size[1]
print("Nombre de Pixels de l'image de base")
print(longueur, largeur)
# on va tronquer limage en 8 par 8
while longueur%8 !=0:
    longueur -=1

while largeur%8 !=0:
    largeur -=1
print("Nombre de Pixels de l'image modulo 8")
print(longueur, largeur) 

box = [0,0,longueur, largeur]
imTronk=image.crop(box)
#imTronk.show()

imTronk.save("montagne2.png","PNG")

#on passe l'image en matrice RGB

M = np.array(imTronk)
print("Image sous forme de Matrice \n",M)
MatriceRouge=M[:,:,0]
MatriceVerte=M[:,:,1]
MatriceBleu=M[:,:,2]
#print(MatriceVerte)

#On centre en 0 les matrices (de -128 a 127)

MatriceRouge=-127.+MatriceRouge
MatriceVerte=-127.+MatriceVerte
MatriceBleu=-127.+MatriceBleu
print("Matrice rouge centré \n",MatriceRouge)

#Matrice de passage P par la méthode DCT2
def CalculP():
    P=np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            if i==0:
                Ci=1/np.sqrt(2)
            else:
                Ci=1
            P[i,j]=(Ci/2)*(np.cos((2*j+1)*i*np.pi/16))
    return P

P=CalculP()
Pinv=np.transpose(CalculP())
print("Matrice P \n",P)
print("on verifie que p*ptransposee donne l'identite")
print("P*P^-1",np.dot(P,Pinv))
#On creer la matrice Q JPEG
Q = np.array([[16,11,10,16,24,40,51,61],[12,12,13,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
print("Matrice Q du format JGPEG\n",Q)
#On divise M en bloc 8x8
def compression(M,f): # f représente le filtre haute frequence
    D=np.zeros(np.shape(M))
    print("hello")
    print((np.shape(M)[0]//8)+1)
    print((np.shape(M)[1]//8)+1)
    for i in range((np.shape(M)[0]//8)):  #on prend les blocs 8x8 
        for j in range((np.shape(M)[1]//8)):
            B=M[i*8:i*8+8,j*8:j*8+8]#on applique le changement le base B=PMP^T
            B=np.dot(np.dot(P,B),Pinv) #on calcule B du bloc 
            B=B/Q
            B=np.trunc(B) #partie entiere 
            for k in range(np.shape(B)[0]):
                for l in range(np.shape(B)[1]):
                    if k+l>=f:
                               B[k,l]=0 #filtrage haute fréquence 
            D[i*8:i*8+8,j*8:j*8+8]=B #on rentre B dans la matrice compressée D
    return D
Drouge=compression(MatriceRouge,10)
Dverte=compression(MatriceVerte,10)
Dbleu=compression(MatriceBleu,10)
print("Matrice RGB sous format compressé")
print(Drouge)
print(Dverte)
print(Dbleu)
#On réassemble la matrice Matrice compressée
MC=np.zeros(np.shape(M))
for i in range(np.shape(MC)[0]):
        for j in range(np.shape(MC)[1]):
            MC[i,j,0]=Drouge[i,j]
            MC[i,j,1]=Dverte[i,j]
            MC[i,j,2]=Dbleu[i,j]
print("Matrice finale compressée")
print(MC)

def compteur(M): #calcul du nombre d'element non nul 
    cmp = 0
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if M[i,j] != 0:
                cmp += 1
    return cmp
print("Nombres d'éléments dans chaque matrice RGB")
print(compteur(Drouge))
print(compteur(Dverte))
print(compteur(Dbleu))

def tauxcompression(M): #calcul du taux de compression 
    taux = compteur(M)/(longueur*largeur)
    return taux
print("Taux de compression de chaque matrice RGB")
print(100-tauxcompression(Drouge)*100)
print(100-tauxcompression(Dverte)*100)
print(100-tauxcompression(Dbleu)*100)

def decompression(M):
    D=np.zeros(np.shape(M))
    for i in range((np.shape(M)[0]//8)):  #on prend les blocs 8x8 
        for j in range((np.shape(M)[1]//8)):
            B=M[i*8:i*8+8,j*8:j*8+8]
            B=B*Q
            B=np.dot(np.dot(Pinv,B),P) #on calcule B du bloc 
            D[i*8:i*8+8,j*8:j*8+8]=B #on rentre B dans la matrice compressée D
    return D
MrougeDC=decompression(Drouge)
MverteDC=decompression(Dverte)
MbleuDC=decompression(Dbleu)
print("Matrice RGB decompressée")
print(MrougeDC)
print(MverteDC)
print(MbleuDC)
#on décentre les valeurs
MrougeDC=+127.+MrougeDC
MverteDC=+127.+MverteDC
MbleuDC=+127.+MbleuDC

#on Tronque les de 0 à 255
for i in range(np.shape(MrougeDC)[0]):
    for j in range(np.shape(MrougeDC)[1]):  
        MrougeDC[i,j]=max(0,(MrougeDC[i,j]))
        MverteDC[i,j]=max(0,(MverteDC[i,j]))
        MbleuDC[i,j]=max(0,MbleuDC[i,j])
        MrougeDC[i,j]=min(255,MrougeDC[i,j])
        MverteDC[i,j]=min(255,MverteDC[i,j])
        MbleuDC[i,j]=min(255,MbleuDC[i,j])
#On réassemble la matrice M de l'image
MDC=np.zeros(np.shape(M))
for i in range(np.shape(MDC)[0]):
        for j in range(np.shape(MDC)[1]):
            MDC[i,j,0]=MrougeDC[i,j]
            MDC[i,j,1]=MverteDC[i,j]
            MDC[i,j,2]=MbleuDC[i,j]
print("Matrice RGB sous forme d'une seule matrice")
print(MDC)
#Comparaison de la matrice originale avec la matrice compressée
print("Comparaison en norme 2 de la matrice originale avec la nouvelle matrice")
print("norm (Mdépart-Mfinale)/norm(Mdépart)")
print(np.linalg.norm(M-MDC)/np.linalg.norm(M))
#on retransforme la matrice en image
imagev2=Image.fromarray(np.uint8(MDC))
imTronk.show()
imagev2.show()

# on compare avec une DTC deja implémenté
MatriceOrdi=dct(dct(M,axis=0,norm='ortho'),axis=1,norm='ortho')
MatriceOrdi=idct(idct(MatriceOrdi,axis=0,norm='ortho'),axis=1,norm='ortho')

#On retransforme la matrice en image
imageOrdi=Image.fromarray(np.uint8(MatriceOrdi))
imageOrdi.show()
