# Данные

`digits8.npz` — рукописные цифры 8×8, 1797 примеров:

- `images` — `uint8`, форма `(1797, 8, 8)`, яркость от 0 до 16;
- `labels` — `uint8`, форма `(1797,)`, цифра от 0 до 9.

Источник: E. Alpaydin, C. Kaynak. *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository, 1998, [doi:10.24432/C50P49](https://doi.org/10.24432/C50P49), лицензия CC BY 4.0. Это тестовая часть набора в том виде, в каком её распространяет scikit-learn (`sklearn.datasets.load_digits`): битовые карты 32×32 разбиты на блоки 4×4, в каждом посчитано число закрашенных точек.

Файл читается так:

```python
with np.load("data/digits8.npz") as f:
    images, labels = f["images"], f["labels"]
```
