# Aygaz-Goruntu-Isleme

Bölüm 1: Gerekli Kütüphanelerin ve Veri Setinin Hazırlanması
Neler Yaptık:
Kütüphanelerin yüklenmesi:

tensorflow, keras, cv2, matplotlib, sklearn gibi kütüphaneleri kullandık.
kagglehub kullanarak, Kaggle üzerindeki bir veri setini indirdik.
Veri setinin yolu:

Kaggle veri setini Google Colab'e bağladık ve kullanılacak dosya dizinlerini ayarladık.
Amaç:
Bu bölümde kullanılan araçlar ve veri seti ayarları ile proje altyapısını kurduk. Veri setini indirerek, hayvan resimlerine erişim sağladık.

Bölüm 2: Veri Seti Hazırlığı
Neler Yaptık:
Hayvan sınıflarının tanımlanması:

Kullanılacak hayvan türlerini (collie, dolphin, elephant, vb.) listeledik.
Resimlerin hazırlanması:

Her hayvan sınıfından en fazla 650 resim seçtik.
Resimler, 128x128 boyutlarına yeniden boyutlandırıldı.
Piksel değerleri normalize edildi (0-255 aralığından 0-1 aralığına çekildi).
Eğitim ve test verilerinin oluşturulması:

Verileri %70 eğitim ve %30 test olacak şekilde bölüp sınıfları kategorik hale getirdik (to_categorical).
Amaç:
Bu bölümde veri setini modele uygun formata dönüştürdük, böylece model hem verimli bir şekilde öğrenebilir hem de doğruluk test edilebilir hale geldi.

Bölüm 3: Veri Artırma (Augmentation)
Neler Yaptık:
ImageDataGenerator kullandık:
Görsellerin döndürülmesi, ölçeklenmesi, kaydırılması gibi yöntemlerle eğitim setini zenginleştirdik.
Bu teknik, modelin öğrenim kapasitesini artırır ve aşırı öğrenmeyi (overfitting) önler.
Amaç:
Veri artırma ile modelin daha çeşitli ve genel bir veri üzerinde öğrenmesini sağladık. Bu, gerçek hayatta karşılaşılabilecek varyasyonlarla daha iyi başa çıkmasına olanak tanır.

Bölüm 4: CNN Modelinin Tasarımı
Neler Yaptık:
Katmanlar ekledik:

Convolutional Layers (Conv2D): Görsellerin özelliklerini (kenarlar, desenler) çıkartır.
MaxPooling Layers: Görsellerin boyutunu küçültürken önemli bilgileri korur.
Flatten Layer: Çok boyutlu çıktıyı tek boyutlu bir forma dönüştürür.
Dense Layer: Özellikleri sınıflara ayırır.
Dropout oranını ayarladık:

Aşırı öğrenmeyi azaltmak için Dropout oranını %30 olarak belirledik.
Modeli derledik:

adam optimizasyon algoritmasını kullandık.
Kayıp fonksiyonu olarak categorical_crossentropy kullandık.
Amaç:
Bu bölümde, hayvan resimlerini doğru şekilde sınıflandırabilecek bir derin öğrenme modeli tasarladık.

Bölüm 5: Modelin Eğitilmesi
Neler Yaptık:
Eğitim işlemi:

fit fonksiyonuyla modeli eğitim verileri üzerinde eğittik.
Veri artırmayı eğitim sırasında kullandık.
Doğrulama işlemi:

Modeli test verileriyle doğrulayarak, eğitim sırasında performansını ölçtük.
Amaç:
Bu adımda modelimiz hayvan türlerini öğrenmeye başladı ve doğrulama verileriyle performansı test edildi.

Bölüm 6: Modelin Test Edilmesi
Neler Yaptık:
Modelin test setindeki performansı:

evaluate fonksiyonunu kullanarak modelin test doğruluğunu ölçtük.
Test doğruluğunu değerlendirme:

Eğer doğruluk oranı düşükse, modelin parametrelerini veya yapısını değiştirme gerekliliğini belirttik.
Amaç:
Bu bölümde, modelin eğitilmemiş veri üzerindeki başarısını kontrol ettik.

Bölüm 7: Resim Manipülasyonu
Neler Yaptık:
Manipülasyon uyguladık:
Test setindeki resimlerin parlaklıklarını artırdık (convertScaleAbs).
Modelin manipüle edilmiş resimler üzerindeki başarısını değerlendirdik.
Amaç:
Modelin, gerçek dünyada karşılaşılabilecek manipüle edilmiş resimlere olan dayanıklılığını ölçtük.

Bölüm 8: Renk Sabitliği Algoritması
Neler Yaptık:
Gray World algoritması:

Görsellerdeki renk kanallarını eşitleyerek, her kanalın ortalamasını diğer kanallarla uyumlu hale getirdik.
Değerlendirme:

Manipüle edilmiş resimler üzerinde renk sabitliği algoritmasını uyguladık ve modeli tekrar test ettik.
Amaç:
Görsellerin renk farklılıklarının sınıflandırmaya olan etkisini inceledik ve modelin daha tutarlı sonuçlar vermesini sağladık.

Bölüm 9: Sonuçların Karşılaştırılması
Neler Yaptık:
Doğruluk karşılaştırması:
Üç farklı senaryoda (Orijinal, Manipüle Edilmiş, Renk Sabitliği) model doğruluk oranlarını ölçtük.
Sonuçları bir bar grafiği ile görselleştirdik.
Amaç:
Modelin farklı test setleri üzerindeki performansını karşılaştırarak, hangi yöntemlerin daha etkili olduğunu gözlemledik.

Genel Sonuç
Bu projede:

Bir veri seti hazırlayıp derin öğrenme modeline uygun hale getirdik.
CNN mimarisiyle bir sınıflandırma modeli tasarladık ve eğittik.
Manipülasyon ve renk sabitliği algoritmalarıyla model dayanıklılığını test ettik.
Sonuçları görselleştirerek modelin genel başarısını değerlendirdik.
Bu adımları geliştirerek doğruluğu daha da artırabilirsiniz (ör. katman sayısını artırma, hiperparametre optimizasyonu, farklı veri artırma teknikleri vb.).
