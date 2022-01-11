# Using CNNs for image classification

# Klasifikátor využívající konvolučních neuronových sítí pro rozpoznání obrazu
- Vzhledem k tomu, že jsem v týmu sám, využil jsem možnost zpracovat jen jednu modalitu - obraz
- Hledaná osoba je tedy detekována pouze pomocí obrazu skrz natrénovanou konvoluční neuronovou síť

## Prostředí
- Běží pod Python3 a využívá moduly uvedené v souboru requirements.txt.
- Instalace potřebných modulů: pip install -r requirements.txt
- Klasifikátor je postaven na modulu Keras s Tensorflow backendem

## Spuštění
- Natrénovaná neuronová síť se momentálně nachází ve složce SRC/modeldata a její výsledky na testovacích
datech v kořenovém adresáři v souboru image_CNN.txt

- Pro nové natrénování a evaluaci výsledku je třeba spustit příkaz 'python SRC/SUR_image.py'.
Nové natrénování a vyhodnocení ovšem vyžaduje, aby se v adresáři SRC nacházely složky 'train' a 'dev',
obě obsahující podadresáře 'target' a 'non_target' s příslušným obsahem.

    - python SRC/SUR_image.py

- Pokud nové trénování není třeba a uživatel chce spustit pouze detekci osoby na fotkách
umístěných v adresáři 'SRC/eval/all', tak skript spouští s parametrem '--notrain' a klasifikátor
vychází z lokálně uložené natrénované neuronové sítě ze složky SRC/modeldata.

    - python SRC/SUR_image.py --notrain

