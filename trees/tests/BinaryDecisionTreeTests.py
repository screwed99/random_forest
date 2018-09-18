import pandas as pd



x = pd.DataFrame([
    (9, 81.0, 73.3),
    (3, 27.0, 37.7),
    (4, 54.0, 33.7)
    ],
    columns=[
        'age',
        'weight',
        'height'])

y = pd.Series([
    0,
    1,
    1],
    name="class")

x_and_y = pd.concat([x['age'], y], axis=1, names=['age', 'class'])

x_sorted_by_age = x_and_y.sort_values(['age'], ascending=[True])
