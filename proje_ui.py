# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:05:01 2020

@author: ayusa
"""
from os import walk
import glob
import threading
import warnings
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *


import cv2
import proje_ml as algo
from PIL import ImageTk, Image
# grafiği arayüze koymak için gerekli
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

warnings.filterwarnings("ignore")
video_name = 0
kayitli_turler = []
egitime_devam_mi = False
tahmin = False
sinif = -2
test_sayisi = 0
sinif1_egitildi_mi = False
sinif2_egitildi_mi = False


class Tur_Sinifi:
    def __init__(self, name):
        self.nesne_turu = name
        self.kayitli_modeller = []
        # self.kayitli_modeller.append(name)

    def model_kaydet(self, yeni_model):
        self.kayitli_modeller.append(yeni_model)

    def getTurIsmi(self):
        return self.nesne_turu

    def getModeller(self):
        return self.kayitli_modeller



class MainWindow:
    def __init__(self, root):
        # global tur
        global siniflar_liste
        global algoritmalar
        self.root = root
        self.root.geometry("1200x600")
        self.root.title("Falling Object Classification")

        self.cap = cv2.VideoCapture(0)

        # KAMERA
        self.panel = Label()
        self.frame = None
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.kamera, args=())
        self.thread.start()


        self.sinif1_test_sayisi = 0
        self.sinif2_test_sayisi = 0

        # ANA EKRAN
        self.mevcut_modeller = Button(root, text="Mevcut Modeller", command=self.mevcut_model_penceresi)
        self.yeni_model_buton = Button(root, text="Yeni Model Eğit", command=self.yeni_model_penceresi)
        self.exit_button = Button(root, text="Çıkış", command=self.cikis, style='W.TButton')

        # MEVCUT MODELLER MENUSU
        # geri butonu
        # combo box
        # radio butonları
        # model seç butonu
        # ÇALIŞTIR BUTONU
        self.r_var = StringVar()
        self.modeller_rd = []
        self.geri_buton = Button(root, text="Geri", command=self.geri_butonu_tiklandi)
        self.mevcut_tur_secin_yazisi = Label(root, text="Kullanılacak Kategoriyi Seçiniz:")
        self.model_listesi_cb = Combobox(root, values=siniflar_liste, state="readonly")
        # self.model_listesi_cb['values'] = tur.tur_getir()
        self.model_listesi_cb.bind("<<ComboboxSelected>>", self.set_radio)
        self.model_sec_buton = Button(root, text="Modeli Seç", command=self.hangi_model_secildi)
        self.modeli_calistir = Button(root, text="Çalıştır", command=self.modeli_calistir_func)
        self.modeli_durdur = Button(root, text="Durdur", command=self.modeli_durdur_fund)
        self.model_info = StringVar()
        self.model_info_label = Label(root, textvariable=self.model_info)
        self.sinif1_bulunan = 0
        self.sinif2_bulunan = 0
        self.bulunanlar_canvas = Canvas(root, bg="#F0F0F0", height=120, width=150)


        # YENİ MODEL EGİT MENUSU
        # geri butonu
        # KATEGORİ SEÇİN YAZISI
        # KATEGORİLER COMBO BOX
        # YENİ KATEGORİ GİRİŞ BUTONU
        # ALGORİTMA İsim Girin yazısı
        # isim alma yeri ve giriş butonu
        # 1. sınıf için yazı, başlat - bitir butonları
        # 2. sınıf için yazı, başlat - bitir butonları
        # Algoritma Seçme Yazısı
        # Algoritmalar Radio butonları
        # ALGORİTMA EĞİT BUTONU
        # KAYDET BUTONU -> anasayfaya git
        self.kategori_secin_yazi = Label(root, text="Kategori Seçiniz:")
        self.model_belirle_cb = Combobox(root, values=siniflar_liste, state="readonly")
        self.yeni_kategori_buton = Button(root, text="Yeni Kategori Ekle", command=self.yeni_kategori_penceresi)
        self.yeni_model_ismi_girin = Label(root, text="Algoritma İsmi Girin:")
        self.algoritma_ismi = StringVar()
        self.yeni_model_ismi_input = Entry(root, textvariable=self.algoritma_ismi)
        self.yeni_model_ismi_giris = Button(root, text="Giriş", command=self.yeni_sinif_ekleme)
        self.sinif1_text = Label(root, text="İlk sınıf için Eğitim: ")
        self.sinif1_baslat = Button(root, text="Başlat", command=lambda: self.egitimi_baslat(True, 1))
        self.sinif1_bitir = Button(root, text='Bitir', command=lambda: self.egitimi_durdur(False, 1))
        self.sinif2_text = Label(root, text="İkinci sınıf için Eğitim: ")
        self.sinif2_baslat = Button(root, text="Başlat", command=lambda: self.egitimi_baslat(True, 2))
        self.sinif2_bitir = Button(root, text='Bitir', command=lambda: self.egitimi_durdur(False, 2))
        self.algoritma_seciniz = Label(root, text="Algoritma Seçiniz:")
        self.algoritma_radios = []
        self.algo_sonuclari = [] # label listesi
        self.algo_isim_variable = StringVar()

        # GRAFIK
        # self.axes = algo.visualization()
        # self.canvas = FigureCanvasTkAgg(self.axes, master=root)
        # self.canvas.draw()
        # self.graph = self.canvas.get_tk_widget()
        self.graph = Label(root)
        self.canvas_sin1_text = ""
        self.canvas_sin2_text = ""

        self.yeni_model_baslat = Button(root, text="Algoritma Eğit", command=self.algoritma_secildi)
        self.algoritma_kaydet_buton = Button(root, text="Kaydet", command=self.yeni_model_kaydet)

        # YENİ KATEGORİ EKLEME SAYFASI
        self.yeni_kategori_ismi = StringVar()
        self.kategori_ismi_girin_yazi = Label(root, text="Kategori İsmi Girin:")
        self.kategori_ismi_entry = Entry(root, textvariable=self.yeni_kategori_ismi)
        self.kategori_ismi_giris_buton = Button(root, text="Giriş", command=self.yeni_kategori_ekle)


        self.AnaEkran()

    def AnaEkran(self):
        self.herseyi_sil()
        self.mevcut_modeller.grid(row=1, column=1, ipady=15)
        self.yeni_model_buton.grid(row=1, column=2, ipady=15)
        self.exit_button.grid(row=9, column=1, sticky=W)

    def kamera(self):
        global egitime_devam_mi, tahmin
        global sinif, sinif1_egitildi_mi, sinif2_egitildi_mi
        global test_sayisi
        try:
            while not self.stopEvent.is_set():
                original_image, test_sayisi = algo.egitimi_baslat_func(egitime_devam_mi, sinif, tahmin)
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGBA)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image=image)

                if sinif == 1 and test_sayisi != 0:
                    self.sinif1_test_sayisi = test_sayisi
                    sinif1_egitildi_mi = True
                elif sinif == 2 and test_sayisi != 0:
                    self.sinif2_test_sayisi = test_sayisi
                    sinif2_egitildi_mi = True

                # bulunanları ekranda göster
                if tahmin:
                    self.sinif1_bulunan, self.sinif2_bulunan = algo.get_bulunanlar()
                    self.table_update()

                self.panel.configure(image=image)
                # self.panel.image = image
                self.panel.grid(row=0, column=0, rowspan=20)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def set_radio(self, event):
        for w in self.modeller_rd:
            w.destroy()
        secilen = self.model_listesi_cb.get()
        # radio_values = tur.model_getir_turegore(secilen)
        index = -1
        for i, t in enumerate(kayitli_turler):
            if secilen == t.getTurIsmi():
                index = i
        radio_values = kayitli_turler[index].getModeller()
        for num, t in enumerate(radio_values, 1):
            b = Radiobutton(root, text=t, variable=self.r_var, value=t)
            b.grid(row=num + 2, column=1)
            self.modeller_rd.append(b)
            index = num

        self.model_sec_buton.grid(row=index + 3, column=1)
        self.modeli_calistir.grid(row=index + 4, column=1)
        self.modeli_durdur.grid(row=index + 5, column=1)

    def mevcut_model_penceresi(self):
        self.herseyi_sil()
        self.geri_buton.grid(row=0, column=1, sticky=W)
        self.mevcut_tur_secin_yazisi.grid(row=1, column=1)
        self.model_listesi_cb.grid(row=2, column=1)


    def yeni_model_penceresi(self):
        self.herseyi_sil()
        self.geri_buton.grid(row=0, column=1, sticky=W)
        self.kategori_secin_yazi.grid(row=1, column=1)
        self.model_belirle_cb.grid(row=2, column=1)
        self.yeni_kategori_buton.grid(row=2, column=2, sticky=E)

        self.yeni_model_ismi_girin.grid(row=3, column=1, columnspan=2)
        self.yeni_model_ismi_input.grid(row=4, column=1)
        self.yeni_model_ismi_giris.grid(row=4, column=2)

    def yeni_sinif_ekleme(self):
        # Tür seçilmişse Algoritma ismini sınıfı içerisine yaz.

        #print("Algoritma ismi: ", self.algoritma_ismi.get(), " kategori: ", self.model_belirle_cb.get())
        if len(self.algoritma_ismi.get()) == 0 or len(self.model_belirle_cb.get()) == 0:
            messagebox.showerror("Eksik Bilgi", "İsim ve Kategori Kısmı Boş Olamaz..")
        else:
            self.sinif1_text.grid(row=5, column=1, columnspan=2)
            self.sinif1_baslat.grid(row=6, column=1)
            self.sinif1_bitir.grid(row=6, column=2)
            self.sinif2_text.grid(row=7, column=1, columnspan=2)
            self.sinif2_baslat.grid(row=8, column=1)
            self.sinif2_bitir.grid(row=8, column=2)
            self.yeni_model_baslat.grid(row=9, column=1, ipadx=20, columnspan=2)
            self.algoritma_seciniz.grid(row=10, column=1, columnspan=2)
            i = 11
            for algo in algoritmalar:
                rb = Radiobutton(root, text=algo, variable=self.algo_isim_variable, value=algo)
                rb.grid(row=i, column=1, columnspan=2, sticky=W)
                self.algoritma_radios.append(rb)
                i += 1


            self.algoritma_kaydet_buton.grid(row=i + 2, column=1, columnspan=2)

    def yeni_kategori_penceresi(self):
        self.herseyi_sil()
        self.geri_buton.grid(row=0, column=1)
        self.kategori_ismi_girin_yazi.grid(row=1, column=1, columnspan=2)
        self.kategori_ismi_entry.grid(row=2, column=1)
        self.kategori_ismi_giris_buton.grid(row=2, column=2)

    def geri_butonu_tiklandi(self):
        self.AnaEkran()

    def yeni_kategori_ekle(self):
        if len(self.yeni_kategori_ismi.get()) == 0:
            messagebox.showerror("Boş İsim", "Lütfen isim giriniz.")
        else:
            tur_ismi = self.yeni_kategori_ismi.get()
            yeni_tur = Tur_Sinifi(tur_ismi)

            siniflar_liste.append(yeni_tur.getTurIsmi())
            self.yeni_kategori_eklendi_mi = True
            # combo boxes update
            self.model_listesi_cb['values'] = siniflar_liste
            self.model_belirle_cb['values'] = siniflar_liste
            self.model_belirle_cb.current(len(kayitli_turler))
            self.yeni_model_penceresi()

    def herseyi_sil(self):
        # ana ekranı sil
        self.mevcut_modeller.grid_remove()
        self.yeni_model_buton.grid_remove()
        self.exit_button.grid_remove()
        # mevcut model ekranı sil
        self.geri_buton.grid_remove()
        self.mevcut_tur_secin_yazisi.grid_remove()
        self.model_sec_buton.grid_remove()
        self.model_listesi_cb.grid_remove()
        for rad in self.modeller_rd:
            rad.destroy()
        self.modeli_calistir.grid_remove()
        self.modeli_durdur.grid_remove()
        self.model_info_label.grid_remove()
        # yeni model ekranı sil
        self.kategori_secin_yazi.grid_remove()
        self.model_belirle_cb.grid_remove()
        self.yeni_kategori_buton.grid_remove()
        self.yeni_model_ismi_girin.grid_remove()
        self.yeni_model_ismi_input.grid_remove()
        self.yeni_model_ismi_giris.grid_remove()
        self.sinif1_text.grid_remove()
        self.sinif1_baslat.grid_remove()
        self.sinif1_bitir.grid_remove()
        self.sinif2_text.grid_remove()
        self.sinif2_baslat.grid_remove()
        self.sinif2_bitir.grid_remove()
        self.algoritma_seciniz.grid_remove()
        for a in self.algoritma_radios:
            a.destroy()
        for lb in self.algo_sonuclari:
            lb.destroy()
        self.yeni_model_baslat.grid_remove()
        self.algoritma_kaydet_buton.grid_remove()
        self.bulunanlar_canvas.grid_remove()
        self.graph.grid_remove()
        # yeni kategori ekleme sayfasını sil
        self.kategori_ismi_girin_yazi.grid_remove()
        self.kategori_ismi_entry.grid_remove()
        self.kategori_ismi_giris_buton.grid_remove()

    def egitimi_baslat(self, devam, sin):
        answer_cont = False

        if sinif1_egitildi_mi and sin == 1:
            msg = "İlk sınıf için " + str(
                self.sinif1_test_sayisi) + " öğe tespit edildi. Bunları Silmek istiyor musunuz?"
            answer_cont = messagebox.askyesno("Tekrar Eğitime Başlansın mı?", message=str(msg))

        if sinif2_egitildi_mi and sin == 2:
            msg = "İkinci sınıf için " + str(
                self.sinif2_test_sayisi) + " öğe tespit edildi. Bunları Silmek istiyor musunuz?"
            answer_cont = messagebox.askyesno("Tekrar Eğitime Başlansın mı?", message=str(msg))
        if answer_cont:
            sin = sin * 100

        if sinif1_egitildi_mi == False or sinif2_egitildi_mi == False:
            global egitime_devam_mi
            global sinif, tahmin
            sinif = sin
            egitime_devam_mi = devam
            tahmin = False

    def egitimi_durdur(self, devam, sin):
        global egitime_devam_mi
        global sinif, tahmin
        sinif = sin
        egitime_devam_mi = devam
        tahmin = False

    def algoritma_secildi(self):
        lr_result, knn_result, svc_result, gaus_result, dt_result = algo.algoritma_egit()
        # print("1 lr res: ", lr_result)
        # print("1 knn res: ", knn_result)
        # print("1 svc res: ", svc_result)
        # print("1 gaus res: ", gaus_result)
        # print("1 dt res: ", dt_result)
        results = [lr_result, knn_result, svc_result, gaus_result, dt_result]
        for i in range(5):
            lb = Label(root, text=str(results[i]))
            lb.grid(row=11+i, column=2)
            self.algo_sonuclari.append(lb)

        axes = algo.visualization()
        canvas = FigureCanvasTkAgg(axes, master=root)
        canvas.draw()
        self.graph = canvas.get_tk_widget()
        self.graph.grid(row=1, column=3, rowspan=20)


    def modeli_durdur_fund(self):
        global egitime_devam_mi
        global sinif, tahmin
        egitime_devam_mi = False
        sinif = 3
        tahmin = False

    def modeli_calistir_func(self):
        global egitime_devam_mi
        global sinif, tahmin
        egitime_devam_mi = True
        sinif = 3
        tahmin = True
        self.bulunanlar_canvas.grid(row=4, column=2, rowspan=5, sticky=W)
        self.bulunanlar_canvas.create_rectangle(15, 20, 65, 90, fill="", outline="blue")
        self.bulunanlar_canvas.create_rectangle(85, 20, 135, 90, fill="", outline="red")
        self.canvas_sin1_text = self.bulunanlar_canvas.create_text(40,  50, text=self.sinif1_bulunan)
        self.canvas_sin2_text = self.bulunanlar_canvas.create_text(110, 50, text=self.sinif2_bulunan)

    def hangi_model_secildi(self):
        self.model_info.set("")
        self.model_info_label.grid_remove()

        if len(self.r_var.get()) == 0:
            messagebox.showerror("Eksik Bilgi", "Model Seçiniz")
        else:
            infolar = algo.modeli_dosyadan_getir(self.model_listesi_cb.get(), self.r_var.get())
            self.model_info.set(infolar)
            self.model_info_label.grid(row=2, column=2)


    def yeni_model_kaydet(self):
        # yeni kategori oluşturulmuşsa -> yeni Tur sınıfı oluştur
        # kayıtlı türlere ekle
        # eski kategorilerden seçilmişse -> tür sınıfını güncelle
        tur_ismi = self.model_belirle_cb.get()
        algo_ismi = self.algoritma_ismi.get()
        # yeni kategori ismi girilMEMİŞSE
        if len(self.yeni_kategori_ismi.get()) == 0:
            for i in range(len(kayitli_turler)):
                if kayitli_turler[i].getTurIsmi() == tur_ismi:
                    kayitli_turler[i].model_kaydet(algo_ismi)
                    break
        # yeni kategori ismi GİRİLMİŞSE
        else:
            yeni_tur_sinif = Tur_Sinifi(tur_ismi)
            yeni_tur_sinif.model_kaydet(algo_ismi)
            kayitli_turler.append(yeni_tur_sinif)

        # dosyaya kaydet
        algo.modeli_dosyaya_kaydet(tur_ismi, algo_ismi, self.algo_isim_variable.get())
        self.AnaEkran()

    def cikis(self):
        global egitime_devam_mi
        global sinif
        egitime_devam_mi = False
        sinif = -99
        self.herseyi_sil()
        self.stopEvent.set()
        self.root.destroy()
        print("ÇIKIŞ")

    def table_update(self):
        self.bulunanlar_canvas.itemconfig(self.canvas_sin1_text, text=self.sinif1_bulunan)
        self.bulunanlar_canvas.itemconfig(self.canvas_sin2_text, text=self.sinif2_bulunan)


if __name__ == '__main__':

    siniflar_liste = set()
    kayitli_turler = []

    # _, _, filenames = next(walk("kayıtlı modeller"))

    for file in glob.glob("kayıtlı modeller/*.txt"):

        tur = file[file.index('\\') + 1:file.index('_')]
        siniflar_liste.add(tur)

        model = file[file.index('_') + 1:file.index('.')]

        t = Tur_Sinifi(tur)

        if not kayitli_turler:
            kayitli_turler.append(t)

        f = False
        for k in kayitli_turler:
            if k.getTurIsmi() == t.getTurIsmi():
                k.model_kaydet(model)
                f = True
        if not f:
            t.model_kaydet(model)
            kayitli_turler.append(t)


    siniflar_liste = list(siniflar_liste)

    algoritmalar = ["Liner Regresyon", "K En Yakın Komşu", "Destek Vektör Makineleri", "Gaussian Naive Bayes", "Karar Ağacı"]


    root = Tk()
    root.configure(bg='#F0F0F0')
    root.grid_columnconfigure(4, minsize=100)
    s = Style()
    s.theme_use('winnative')
    s.configure('.', font=('Helvetica', 10, 'bold'))
    gui = MainWindow(root)
    root.mainloop()
