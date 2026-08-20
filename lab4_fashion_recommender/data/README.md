# Данные

Эта папка предназначена для локальной копии [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). Изображения не входят в Git: архив занимает около 572 МБ.

После распаковки ожидается структура:

```text
data/fashion/
├── styles.csv
└── images/
    ├── 10000.jpg
    └── ...
```

Если настроен Kaggle API, загрузка выполняется так:

```powershell
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/fashion --unzip
```
