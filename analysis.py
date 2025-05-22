import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np

separator = '=' * 50

url = 'Datasets/GitHub/ajinmathew/sqliv2.csv'
ajinmathew_2 = pd.read_csv(url, encoding='utf-16le')

ajinmathew_2['Sentence'] = ajinmathew_2['Sentence'].fillna('') # fillna - заменяет NaN на ''
ajinmathew_2['Label'] = ajinmathew_2['Label'].fillna(0)
print()
print(f"размер датасета до удадения дубликатов и пустых строк: {ajinmathew_2.shape}")
ajinmathew_2_cleaned = ajinmathew_2.drop_duplicates()
ajinmathew_2_cleaned = ajinmathew_2_cleaned[ajinmathew_2_cleaned['Sentence'].str.strip() != '']

print(f"размер датасета после удадения дубликатов и пустых строк: {ajinmathew_2_cleaned.shape}")

print(separator)
print('                  ajinmathew_2')
print(separator)
print(ajinmathew_2_cleaned.columns.tolist())
print()
print(ajinmathew_2_cleaned['Label'].value_counts())
print()

sql_querise = ajinmathew_2_cleaned[ajinmathew_2_cleaned['Label'] == 1]['Sentence']
patterns = ["'","--","/*","UNION","OR 1=1","DROP","EXEC","XP_"]
for pattern in patterns:
    count = sql_querise.str.contains(pattern,case=False).sum()
    print(f"Паттерн  '{pattern}': {count} запросов" )
print()
print()

# Длина запроса
ajinmathew_2_cleaned['length'] = ajinmathew_2_cleaned['Sentence'].apply(len)
# Количество специальных символов
ajinmathew_2_cleaned['quotes_count'] = ajinmathew_2_cleaned['Sentence'].str.count("'")
ajinmathew_2_cleaned['comment_count'] = ajinmathew_2_cleaned['Sentence'].str.count("--|#")
ajinmathew_2_cleaned['has_union'] = ajinmathew_2_cleaned['Sentence'].str.contains("UNION", case=False).astype(int)
ajinmathew_2_cleaned['has_or'] = ajinmathew_2_cleaned['Sentence'].str.contains("OR 1=1", case=False).astype(int)

# Пример преобразованных данных
print(ajinmathew_2_cleaned[['Sentence', 'Label', 'length', 'has_union']].head())

import re

# Комбинированное регулярное выражение
pattern = re.compile(
    r'('
    r'\b(OR|AND)\s+[\w\'"]+\s*=\s*[\w\'"]+\s*(--|#|/\*)|'  # OR 1=1 --
    r'=\s*[\w\'"]+\s*\(|'                                   # =SUBSTRING(...)
    r'=\s*(SELECT|UNION|NULL|IF|EXEC)|'                     # =SELECT
    r'=\s*\d+\s*=\s*\d+'                                    # =1=1
    r')',
    re.IGNORECASE
)

# Поиск опасных '='
ajinmathew_2_cleaned['is_sqli'] = ajinmathew_2_cleaned['Sentence'].apply(lambda x: bool(re.search(pattern, str(x)) if pd.notna(x) else False))

# Фильтрация и проверка
sqli_samples = ajinmathew_2_cleaned[ajinmathew_2_cleaned['is_sqli']].sample(5, random_state=42) #.sample - случайным образом выбирает 5 примеров
print("Найдено SQLi с '=':",ajinmathew_2_cleaned['is_sqli'].sum())
print("\nПримеры:")
for query in sqli_samples['Sentence']:
    print("-", query)

print(separator)

# Загружаем датасет SQL-запросов
url = 'Datasets/Kaggle/Modified_SQL_Dataset.csv'
Kaggle = pd.read_csv(url)

# Выводим размер датасета до удаления дубликатов
print(f"размер датасета до удадения дубликатов : {Kaggle.shape}")

# Удаляем дубликаты строк и пустые SQL-запросы
Kaggle_cleaned = Kaggle.drop_duplicates()
Kaggle_cleaned = Kaggle_cleaned[Kaggle_cleaned['Query'].str.strip() != '']


# Удаление пустых строк
Kaggle_cleaned= Kaggle_cleaned[Kaggle_cleaned['Query'].str.strip() != '']
print(f"размер датасета после удадения дубликатов и пустых строк : {Kaggle_cleaned.shape}")


print(separator)
print('                 SQLi From  Kaggel')
print(separator)
print(Kaggle_cleaned.columns.tolist())
print()


# Анализируем распределение меток (безопасные vs вредоносные запросы)
print(Kaggle_cleaned['Label'].value_counts())

print()

# Создаём список паттернов, характерных для SQL-инъекций
patterns = ["'","--","/*","UNION","OR 1=1","DROP","EXEC","XP_", " = "]

# Вычисляем количество SQL-запросов, содержащих указанные паттерны
sql_querise = Kaggle_cleaned[Kaggle_cleaned['Label'] == 1]['Query']
for pattern in patterns:
    count = sql_querise.str.contains(pattern,case=False).sum()
    print(f"Паттерн  '{pattern}': {count} запросов" )


print()
print()


# Создаём новые признаки для анализа SQL-инъекций
Kaggle_cleaned['length'] = Kaggle_cleaned['Query'].apply(len) # Длина запроса
Kaggle_cleaned['quotes_count'] = Kaggle_cleaned['Query'].str.count("'")  # Количество кавычек
Kaggle_cleaned['comment_count'] = Kaggle_cleaned['Query'].str.count("--|#")  # Количество комментариев (-- или #)
Kaggle_cleaned['has_union'] = Kaggle_cleaned['Query'].str.contains("UNION", case=False).astype(int)  # Признак UNION
Kaggle_cleaned['has_or'] = Kaggle_cleaned['Query'].str.contains("OR 1=1", case=False).astype(int)  # Признак OR 1=1

 # Выводим преобразованные данные с новыми признаками
print(Kaggle_cleaned[['Query', 'Label', 'length', 'has_union']].head())


# Комбинированное регулярное выражение для обнаружения SQL-инъекций

pattern = re.compile(
    r'('
    r'\b(OR|AND)\s+[\w\'"]+\s*=\s*[\w\'"]+\s*(--|#|/\*)|'  # OR 1=1 --
    r'=\s*[\w\'"]+\s*\(|'                                   # =SUBSTRING(...)
    r'=\s*(SELECT|UNION|NULL|IF|EXEC)|'                     # =SELECT
    r'=\s*\d+\s*=\s*\d+'                                    # =1=1
    r')',
    re.IGNORECASE
)

#Определяем, содержит ли SQL-запрос паттерны SQL-инъекций
Kaggle_cleaned['is_sqli'] = Kaggle_cleaned['Query'].apply(lambda x: bool(re.search(pattern, str(x)) if pd.notna(x) else False))

# Фильтруем и выбираем примеры SQL-инъекционных запросов
sqli_samples = Kaggle_cleaned[Kaggle_cleaned['is_sqli']].sample(5, random_state=42)
print("Найдено SQLi с '=':", Kaggle_cleaned['is_sqli'].sum())
print("\nПримеры:")
for query in sqli_samples['Query']:
    print("-", query)

print(separator)



# Данные распределения классов
labels_ajinmathew = ajinmathew_2_cleaned['Label'].value_counts()
labels_kaggle = Kaggle_cleaned['Label'].value_counts()

# Создание фигуры и двух графиков
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


#График  Распределение классов: а – SQL Injection Dataset от SAJID576 с платформы Kaggle; б – SQL-data от ajinmathew с площадки GitHub
# График для "а"
axes[0].bar(labels_ajinmathew.index, labels_ajinmathew.values, color=['#d2b48c', '#3498db'], alpha=0.5)  # Бежевый и синий
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Безопасные', 'SQLi'])
axes[0].set_ylabel('Количество запросов')
axes[0].set_title('Распределение классов')
axes[0].text(0.5, -max(labels_ajinmathew.values) * 0.15, 'а', ha='center', fontsize=12, color='gray')

# График для "б"
axes[1].bar(labels_kaggle.index, labels_kaggle.values, color=['#f4a460', '#2ecc71'], alpha=0.5)  # Светло-коричневый и зеленый
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Безопасные', 'SQLi'])
axes[1].set_ylabel('Количество запросов')
axes[1].set_title('Распределение классов')
axes[1].text(0.5, -max(labels_kaggle.values) * 0.15, 'б', ha='center', fontsize=12, color='gray')

# Улучшение визуализации
plt.tight_layout()
plt.show()


# График количества запросов, содержащих паттерны характерных для SQL-инъекций:

# Обновленный список паттернов
patterns = ["'", "--", "/*", "UNION", "DROP", "EXEC", "XP_", "="]

# Подсчёт количества запросов с паттернами
pattern_counts_ajinmathew = [ajinmathew_2_cleaned['Sentence'].str.contains(pattern, case=False).sum() for pattern in patterns]
pattern_counts_kaggle = [Kaggle_cleaned['Query'].str.contains(pattern, case=False).sum() for pattern in patterns]

# Создание фигуры и двух графиков
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Увеличенный размер для лучшего масштаба

# График для "а"
axes[0].bar(patterns, pattern_counts_ajinmathew, color=['#d2b48c'] * len(patterns), alpha=0.6)  # Бежевый с небольшой прозрачностью
axes[0].set_ylabel('Количество запросов')
axes[0].set_title('Запросы с паттернами SQLi')
axes[0].text(len(patterns) / 2, -max(pattern_counts_ajinmathew) * 0.15, 'а', ha='center', fontsize=12, color='gray')
axes[0].tick_params(axis='x', rotation=45)

# График для "б"
axes[1].bar(patterns, pattern_counts_kaggle, color=['#f4a460'] * len(patterns), alpha=0.6)  # Светло-коричневый с небольшой прозрачностью
axes[1].set_ylabel('Количество запросов')
axes[1].set_title('Запросы с паттернами SQLi')
axes[1].text(len(patterns) / 2, -max(pattern_counts_kaggle) * 0.15, 'б', ha='center', fontsize=12, color='gray')
axes[1].tick_params(axis='x', rotation=45)

# Улучшение визуализации
plt.tight_layout()
plt.show()


# Получаем данные о длине запросов
normal_queries_ajinmathew = ajinmathew_2_cleaned[ajinmathew_2_cleaned['Label'] == 0]['length']
anomalous_queries_ajinmathew = ajinmathew_2_cleaned[ajinmathew_2_cleaned['Label'] == 1]['length']

normal_queries_kaggle = Kaggle_cleaned[Kaggle_cleaned['Label'] == 0]['length']
anomalous_queries_kaggle = Kaggle_cleaned[Kaggle_cleaned['Label'] == 1]['length']

# Создание фигуры и двух графиков
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Биннингуем данные
bins = np.histogram_bin_edges(normal_queries_ajinmathew, bins=30)

# Гистограмма для "а" (аномальные наложены на нормальные)
axes[0].hist(normal_queries_ajinmathew, bins=bins, color='#ffcc99', alpha=0.7, edgecolor='black', label="Нормальные")
axes[0].hist(anomalous_queries_ajinmathew, bins=bins, color='#ff6f61', alpha=0.5, edgecolor='black', label="Аномальные")
axes[0].set_xlabel('Длина запроса')
axes[0].set_ylabel('Количество запросов')
axes[0].set_title('Распределение длины запросов (ajinmathew)')
axes[0].legend()
axes[0].text(normal_queries_ajinmathew.max() * 0.7, -len(normal_queries_ajinmathew) * 0.02, 'а', ha='center', fontsize=12, color='gray')

# Биннингуем данные для второго графика
bins_kaggle = np.histogram_bin_edges(normal_queries_kaggle, bins=30)

# Гистограмма для "б" (аномальные наложены на нормальные)
axes[1].hist(normal_queries_kaggle, bins=bins_kaggle, color='#ffdb99', alpha=0.7, edgecolor='black', label="Нормальные")
axes[1].hist(anomalous_queries_kaggle, bins=bins_kaggle, color='#ff8855', alpha=0.5, edgecolor='black', label="Аномальные")
axes[1].set_xlabel('Длина запроса')
axes[1].set_ylabel('Количество запросов')
axes[1].set_title('Распределение длины запросов (Kaggle)')
axes[1].legend()
axes[1].text(normal_queries_kaggle.max() * 0.7, -len(normal_queries_kaggle) * 0.02, 'б', ha='center', fontsize=12, color='gray')

# Улучшение визуализации
plt.tight_layout()
plt.show()
