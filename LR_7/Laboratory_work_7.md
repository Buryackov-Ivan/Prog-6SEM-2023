## Лабораторная Работа 7

**Задание выполнил:** Буряков Иван Олегович ИВТ(1)

### Задание: 
Видеозапись занятия: https://disk.yandex.ru/d/FQQQbiD8FpIzBg



Выполнить с использованием Google Colab (https://colab.research.google.com/).

Создайте документ и разместите в нём код, находящийся внизу. Загрузите файл с данными о ценах на недвижимость в колаб. 

На основе кода ниже и данных, размещенных выше, реализуйте:

1. Визуализацию данных. 
2. Линейную модель, которая учитывала бы только размер жилья.
3. Полиномиальную модель (степени 2 и 3), учитывающую только размер жилья.
4. Линейную модель (с помощью scikit-learn), которая учитывала бы размер жилья и количество комнат (см. Примечание 1)
5. Предскажите значения для двух объектах недвижимости с использованием этих трех моделей: 1650,3; 2200,4.
6. В ответе к лабораторной работе и в колабе представьте предсказанные значения стоимости объектов недвижимости для всех построенных моделей.
7. Оцените ошибку для созданных моделей. Опишите какая ошибка больше, а какая меньше и укажите причину.


```
%%capture
!wget https://gist.githubusercontent.com/nzhukov/3f5d37624c0cdce27f19cf7dad8fd29a/raw/7d3cba39872ee086c698e1fa2b283c45d064979d/ex1data2.txt  # этот пункт можно не выполнять,
# данные лежат в Moodle: Источник данных ИСР 1.3.
```


[Источник данных ИСР 1.3.](https://github.com/Buryackov-Ivan/Prog-6SEM-2023/blob/main/LR_7/%D0%98%D1%81%D1%82%D0%BE%D1%87%D0%BD%D0%B8%D0%BA%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%98%D0%A1%D0%A0%201.3.txt)


```
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")
x = data[:,0]
y = data[:,1]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]


f1p, residuals, rank, sv, rcond = np.polyfit(x, y, 1, full=True)
f1 = np.poly1d(f1p)
fx = np.linspace(0,x[-1],500) 

plt.scatter(x, y, s=10)
plt.plot(fx,f1(fx),linewidth=1.0,color='r')
plt.title('Трафик веб-сайта за последний месяц')
plt.xlabel("время")
plt.ylabel("запросы/час")
plt.xticks([w*7*24 for w in range(10)],
           ["неделя %i" % w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle="-", color='0.8')
plt.show()
```

**Примечание 1**

Следует заметить, что метод polyfit подходит только для создания модели, предсказывающей значение целевого параметра по **одной** переменной (мы используем размер дома). Однако, в этом пункте требуется создать модель, которая будет учитывать **и размер дома, и количество комнат**.  Поэтому polyfit нам не подходит. 

Для выполнения этого пункта следует использовать пакет scikit learn и оттуда модуль [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression). Здесь может возникнуть путаница в понятиях "полиномиальная" модель находится в блоке LinearRegression? Да, "полиномиальность" здесь - просто характеристика целевой функции модели. При этом модель всё равно предполагает, что целевой параметр и фичи модели **линейно** связаны — поэтому «линейная».

По ссылкам ниже вы найдете примеры использование этой модели: 

* https://thecode.media/](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
* https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
* http://espressocode.top/python-implementation-of-polynomial-regression/


### Решение:


URL: [Лабораторная работа 7](https://replit.com/@Buryackov-Ivan/6SEM-LR3?migrateNonNix=1)
