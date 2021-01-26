# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:17:47 2020

@author: ayusa
"""

"""
Kendi modelimizi eğiteceğimiz kısım burası.
kullanıcı başlat dediği zaman buradan kamera açılacak. 
"""
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#import sklearn.external.joblib as extjoblib
import joblib   # model kaydet
import pandas as pd
import re   # parantez kaldırmak için

# kullanılan modeller
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector # feature selection
# cross validation
from sklearn.model_selection import train_test_split

video_name = 0

threshold = 298
foods = []
temp_foods = []
tekrar_test_yapildi_mi = False
eskileri_sildik_mi = False
ID = 0

# RGB değil BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
rose_dust = (111, 94, 158)
cyan = (255, 153, 51)
black = [0, 0, 0]

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

foods_colors = []
areas = []
sinif1_test_sayisi = 0
sinif2_test_sayisi = 0

kullanilan_model = None
kullanilan_feature = []

# video özellikleri
cap = cv.VideoCapture(video_name, cv.CAP_DSHOW)


global original_image

egitim_baslat = True


knn_model = KNeighborsClassifier(n_neighbors=3)
lr_model = LogisticRegression(solver="liblinear")
svc_model = SVC(kernel="linear")
gaus_model = GaussianNB()
dt_model = DecisionTreeClassifier()

lr_result = 0
knn_result = 0
svc_result = 0
gaus_result = 0
dt_result = 0
lr_fetures = []
knn_fetures = []
svc_fetures = []
gaus_fetures = []
dt_fetures = []


x_train = [0, 0]
sinif1_bulunan = 0
sinif2_bulunan = 0



def visualization() -> Figure:
    global x_train
    fig = Figure(figsize=(5, 5))

    y1 = np.zeros(sinif1_test_sayisi, dtype=int)
    y2 = np.ones(sinif2_test_sayisi, dtype=int)
    y = np.concatenate((y1, y2))

    ax = fig.subplots()
    sns.scatterplot(x=x_train[0], y=x_train[1], hue=y, ax=ax)
    return fig


class food:
    def __init__(self, x, y, w, h, ID, image):
        self.apricot = []
        self.minx = x
        self.miny = y
        self.maxx = w
        self.maxy = h
        self.ID = ID
        self.center_x = (self.minx + self.maxx) // 2
        self.center_y = (self.miny + self.maxy) // 2
        self.apricot.append([self.minx, self.miny, self.maxx, self.maxy, self.center_x, self.center_y])
        self.size = (self.maxx - self.minx) * (self.maxy - self.miny)
        self.blue_values = []
        self.green_values = []
        self.red_values = []

        self.extract_pixels(image)

    def uzunluk(self):
        return len(self.apricot)

    def extract_pixels(self, image):
        global foods_colors
        red_pixels = []
        green_pixels = []
        blue_pixels = []

        ori_img = original_image[self.miny:self.maxy, self.minx: self.maxx]

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if not image[x, y] == 0:  # apply varsa
                    red_pixels.append(ori_img[x, y, 2])
                    green_pixels.append(ori_img[x, y, 1])
                    blue_pixels.append(ori_img[x, y, 0])

        red_sort = np.sort(red_pixels)
        min_red = red_sort[0]
        red_genislik = (red_sort[-1] - min_red) / 10

        # Kirmizi - Red
        i = 1
        redler = []
        reds = []
        for r in red_sort:
            if min_red + (red_genislik * i) >= r:
                reds.append(r)
            else:
                redler.append(reds)
                i += 1
                reds = []
        # son listeyide ekle
        redler.append(reds)
        # self.red_values = redler

        for rm in redler:
            self.red_values.append(np.mean(rm))

        # Yesil - Green
        green_sort = np.sort(green_pixels)
        min_green = green_sort[0]
        green_genislik = (green_sort[-1] - min_green) / 10
        i = 1
        greenler = []
        greens = []
        for g in green_sort:
            if min_green + (green_genislik * i) >= g:
                greens.append(g)
            else:
                greenler.append(greens)
                i += 1
                greens = []

        # son listeyide ekle
        greenler.append(greens)
        # self.green_values = greenler
        for gm in greenler:
            self.green_values.append(np.mean(gm))

        # Mavi - Blue
        blue_sort = np.sort(blue_pixels)
        min_blue = blue_sort[0]
        blue_genislik = (blue_sort[-1] - min_blue) / 10
        i = 1
        bluelar = []
        blues = []
        for b in blue_sort:
            if min_blue + (blue_genislik * i) >= b:
                blues.append(b)
            else:
                bluelar.append(blues)
                i += 1
                blues = []

        # son listeyide ekle
        bluelar.append(blues)
        # self.blue_values = bluelar
        for bm in bluelar:
            self.blue_values.append(np.mean(bm))

    def add(self, x, y, w, h):
        center_x = (x + w) // 2
        center_y = (y + h) // 2
        self.apricot.append([x, y, w, h, center_x, center_y])

    def isNear(self, x, y, miny_p):  # center_x - center_y - minY

        n = (x, y)
        n = np.array(n)
        found = False
        # bir kayısının x (yatay) boyutu kadar kenarlara daha bakıyoruz.
        img_threshold = self.maxy - self.miny

        # bir obje geldikten sonra aynı hizadan başka obje gelebilir sonra
        if y < self.apricot[-1][5]:
            return False, -1

        for v in self.apricot:
            v = np.array(v)
            minx = v[0]
            maxx = v[2]
            center_x = v[4]
            center_y = v[5]
            if miny_p > center_y and x < maxx + img_threshold and x > minx - img_threshold:  # y-maxy < threshold --- (y > maxy) and
                found = True

        if found:
            return True, self.ID
        return False, -1

    def get_values(self):
        vals = []
        if kullanilan_feature[0] < 10:
            vals.append(self.red_values[kullanilan_feature[0]])
        elif kullanilan_feature[0] < 20:
            vals.append(self.green_values[kullanilan_feature[0]])
        else:
            vals.append(self.blue_values[kullanilan_feature[0]])

        if kullanilan_feature[1] < 10:
            vals.append(self.red_values[kullanilan_feature[1]])
        elif kullanilan_feature[1] < 20:
            vals.append(self.green_values[kullanilan_feature[1]])
        else:
            vals.append(self.blue_values[kullanilan_feature[1]])
        vals = pd.DataFrame(vals).T
        return vals

# kaydet butonuna basınca modeli dosyaya kayıt et.
def modeli_dosyaya_kaydet(tur, model_ismi, algo):
    isim = "Kayıtlı Modeller/" + tur + "_" + model_ismi

    algoritmalar = ["Liner Regresyon", "K En Yakın Komşu", "Destek Vektör Makineleri", "Gaussian Naive Bayes", "Karar Ağacı"]

    index = algoritmalar.index(algo)
    sonuc = 0
    if index == 0:
        kullanilan_model = lr_model
        sonuc = lr_result
        kullanilan_feature = lr_fetures
    elif index == 1:
        kullanilan_model = knn_model
        sonuc = knn_result
        kullanilan_feature = knn_fetures
    elif index == 2:
        kullanilan_model = svc_model
        sonuc = svc_result
        kullanilan_feature = svc_fetures
    elif index == 3:
        kullanilan_model = gaus_model
        sonuc = gaus_result
        kullanilan_feature = gaus_fetures
    elif index == 4:
        kullanilan_model = dt_model
        sonuc = dt_result
        kullanilan_feature = dt_fetures

    joblib.dump(kullanilan_model, isim + ".pkl")


    # modelin özelliklerini txt dosyasına yaz.
    f = open(isim + ".txt", "w", encoding='utf8')
    f.write("Test sayısı {}\n".format(sinif1_test_sayisi + sinif2_test_sayisi))
    f.write("Doğruluk Yüzdesi: {}\n".format(sonuc))
    f.write("Algoritma: {}\n".format(algo))
    f.write("{}\n".format(kullanilan_feature))
    f.close()
    print("Model Kayıt edildi")

# modeli seç dediği zaman dosyadan seçilen modeli getir. kullanilan_model e kaydet.
def modeli_dosyadan_getir(tur, model_ismi):
    global kullanilan_model,kullanilan_feature
    isim = "Kayıtlı Modeller/" + tur + "_" + model_ismi
    kullanilan_model = joblib.load(isim + ".pkl")

    r = open(isim+".txt", "r",  encoding='utf8')
    #info = r.read()     # 'Test sayısı: 40\nDoğruluk Yüzdesi: 95\nAlgoritma: KNN'
    info = ""
    for i in range(3):
        info += r.readline()
    son_satir = r.readline()
    son_satir = son_satir.replace("\n", "")     # newline sil
    son_satir = son_satir.strip('][').split(', ')   # parantezleri sil
    son_satir = list(map(int, son_satir))       # list yap

    kullanilan_feature = son_satir

    r.close()
    return info


# algoritma ismine göre objesini oluştur.
def algoritma_egit():
    global eskileri_sildik_mi, temp_foods, foods, x_train
    global lr_result, knn_result, svc_result, gaus_result, dt_result
    global lr_fetures, knn_fetures, svc_fetures, gaus_fetures, dt_fetures

    # foods içinde değerler kesin var.
    x = []
    for f in foods:
        x.append(np.concatenate((f.red_values, f.green_values, f.blue_values), axis=0))

    # eğer tekrar eğitim yaptıysak temp_foods içinde de değerler olacak.
    if eskileri_sildik_mi:
        for f in temp_foods:
            x.append(np.concatenate((f.red_values, f.green_values, f.blue_values), axis=0))

    x = pd.DataFrame(x)
    x.fillna(method='ffill', axis=1, inplace=True)

    y1 = np.zeros(sinif1_test_sayisi, dtype=int)
    y2 = np.ones (sinif2_test_sayisi, dtype=int)
    y = np.concatenate((y1, y2))


    # feature selection
    lr_fetures_1 = featureSelection(lr_model, x, y)
    knn_fetures_1 = featureSelection(knn_model, x, y)
    svc_fetures_1 = featureSelection(svc_model, x, y)
    gaus_fetures_1 = featureSelection(dt_model, x, y)
    dt_fetures_1 = featureSelection(dt_model, x, y)

    # return tuple but we need list
    lr_fetures = list(lr_fetures_1)
    knn_fetures = list(knn_fetures_1)
    svc_fetures = list(svc_fetures_1)
    gaus_fetures = list(gaus_fetures_1)
    dt_fetures = list(dt_fetures_1)

    # print("lr fet: ", lr_fetures)
    # print("knn fet: ", knn_fetures)
    # print("svc fet: ", svc_fetures)
    # print("gaus fet: ", gaus_fetures)
    # print("dt fet: ", dt_fetures)

    # bulduumuz özellikleri ayrı algoritmalar için birleştir
    # hepsi farklı feature seçiyor. farklı dataset verilmeli
    x_lr = pd.concat([x.iloc[:,lr_fetures[0]], x.iloc[:,lr_fetures[1]]], sort=False, ignore_index=True, axis=1)
    x_knn = pd.concat([x.iloc[:,knn_fetures[0]], x.iloc[:,knn_fetures[1]]], sort=False, ignore_index=True, axis=1)
    x_svc = pd.concat([x.iloc[:,svc_fetures[0]], x.iloc[:,svc_fetures[1]]], sort=False, ignore_index=True, axis=1)
    x_gaus = pd.concat([x.iloc[:,gaus_fetures[0]], x.iloc[:,gaus_fetures[1]]], sort=False, ignore_index=True, axis=1)
    x_dt = pd.concat([x.iloc[:,dt_fetures[0]], x.iloc[:,dt_fetures[1]]], sort=False, ignore_index=True, axis=1)


    # her model için kendine ait çıkardığımız featurelar için train test split yapıyoruz.
    # ve bunlarlar modeli eğitiyoruz. iki farklı eğitimin ortalamasını alıyoruz.
    x1, x2, y1, y2 = train_test_split(x_lr, y, test_size=0.7)
    lr_model.fit(x1, y1)
    lr_result = lr_model.score(x2, y2) * 100

    x1, x2, y1, y2 = train_test_split(x_knn, y, test_size=0.7)
    knn_model.fit(x1, y1)
    knn_result = knn_model.score(x2, y2) * 100

    x1, x2, y1, y2 = train_test_split(x_svc, y, test_size=0.7)
    svc_model.fit(x1, y1)
    svc_result = svc_model.score(x2, y2) * 100

    x1, x2, y1, y2 = train_test_split(x_gaus, y, test_size=0.7)
    gaus_model.fit(x1, y1)
    gaus_result = gaus_model.score(x2, y2) * 100

    x1, x2, y1, y2 = train_test_split(x_dt, y, test_size=0.7)
    dt_model.fit(x1, y1)
    dt_result = dt_model.score(x2, y2) * 100


    x_train = x_lr
    # x_train.append(x[knn_fetures])
    # x_train.append(x[svc_fetures])
    # x_train.append(x[gaus_fetures])
    # x_train.append(x[dt_fetures])

    # print("lr res: ", lr_result)
    # print("knn res: ", knn_result)
    # print("svc res: ", svc_result)
    # print("gaus res: ", gaus_result)
    # print("dt res: ", dt_result)


    return lr_result, knn_result, svc_result, gaus_result, dt_result




def cross_val(model, x, y):
    kf = KFold(n_splits=2, random_state=5, shuffle=True)
    result = cross_val_score(model, x, y, cv=kf)
    return result.mean()

def featureSelection(model,x,y):
    feature_select = SequentialFeatureSelector(model,
                                               k_features=2,
                                               forward=True,
                                               floating=False,
                                               scoring='roc_auc',
                                               cv=2)
    feature_select.fit(x, y)
    return feature_select.k_feature_names_


# global class_counter
class_counter = 1

def eskileri_sil(sinif):
    #sinif /= 100
    global eskileri_sildik_mi, temp_foods, foods
    eskileri_sildik_mi = True

    # ilk sınıfı tekrar kaydet
    # ikinciyi yedekle. komple sil.
    if sinif == 1:
        #print(f"birinciyi tekrar kaydet.", foods[0].red_mean)
        temp_foods = foods[sinif1_test_sayisi:]
        #del foods
        foods = []

    # ikinci sinifi tekrar kaydet
    # birinciyi yedekle. komple sil.
    elif sinif == 2:
        #print(f"ikinciyi tekrar kaydet. {len(foods)}")
        temp_foods = foods[:sinif1_test_sayisi]
        #del foods
        foods = []


def egitimi_baslat_func(devam, sinif, tahmin):
    global sinif1_test_sayisi, class_counter
    global sinif2_test_sayisi, eskileri_sildik_mi
    global cap
    global original_image
    test_sayisi = 0

    _, original_image = cap.read()


    if (sinif == 100 or sinif == 200) and eskileri_sildik_mi == False:
        sinif /= 100
        eskileri_sil(sinif)

    if devam:
        original_image = model_egitimi(original_image, tahmin)
    else:
        if class_counter == 1 and sinif == 1:
            sinif1_test_sayisi = len(foods)
            test_sayisi = sinif1_test_sayisi
            print("1 sinif için Tespit Edilen:", sinif1_test_sayisi)

            class_counter += 1
            eskileri_sildik_mi = False

        elif class_counter == 2 and sinif == 2:
            sinif2_test_sayisi = len(foods) - sinif1_test_sayisi
            print("2 sinif için Tespit Edilen:", sinif2_test_sayisi)
            test_sayisi = sinif2_test_sayisi
            class_counter += 1
            eskileri_sildik_mi = False

    return original_image, test_sayisi





# TEK FRAME ÜZERİNDEN GÖRÜNTÜ İŞLEME - NESNE TESPİTİ YAPAR
def model_egitimi(frame, tahmin):
    global ID, sinif2_bulunan, sinif1_bulunan
    renk = green

    tresh = fgbg.apply(frame)

    countour, _ = cv.findContours(tresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in countour:
        if cv.contourArea(i) < 300:
            continue
        areas.append(cv.contourArea(i))

        x1, y1, w, h = cv.boundingRect(i)

        x2 = x1 + w
        y2 = y1 + h
        center_x = (x1 + x1 + w) // 2
        center_y = (y1 + y1 + h) // 2

        found = False

        foo_id = ID
        for f in reversed(foods):
            t, foo_id = f.isNear(center_x, center_y, y1)
            if t:
                f.add(x1, y1, x2, y2)
                found = True
                # burada eğitimi yap.
                if tahmin and f.uzunluk() == 2:
                    x_test = f.get_values()
                    y_pred = kullanilan_model.predict(x_test)
                    #print("Y_pred: ", y_pred)
                    if y_pred == 0:
                        renk = rose_dust
                        sinif1_bulunan += 1
                        #print("Sınıf 1 Tespit: ", sinif1_bulunan)
                    else:
                        renk = yellow
                        sinif2_bulunan += 1
                        #print("Sınıf 2 Tespit: ", sinif2_bulunan)
                break

        if not found:
            ID += 1
            # 			print("NOT FOUND! Add this:",center_x,center_y)
            foo = food(x1, y1, x2, y2, ID, tresh[y1:y2, x1:x2])  # [y1:y2, x1:x2]
            foods.append(foo)
            foo_id = ID



        # cv.drawContours(original_image, i, -1, (0, 255, 255), 3)
        cv.circle(frame, (center_x, center_y), 1, red, 5)  # center
        cv.rectangle(frame, (x1, y1), (x2, y2), renk, 4)

        str_id = "ID: " + str(foo_id)
        cv.putText(frame, str_id, (center_x - 15, center_y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2)

    return frame


def get_bulunanlar():
    global sinif1_bulunan, sinif2_bulunan
    return sinif1_bulunan, sinif2_bulunan