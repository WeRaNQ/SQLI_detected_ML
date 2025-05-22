import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Загружаем датасет с SQL-запросами
url = 'Datasets/Kaggle/Modified_SQL_Dataset.csv'
Kaggle = pd.read_csv(url)

# Удаляем дубликаты строк и пустые SQL-запросы
Kaggle_cleaned = Kaggle.drop_duplicates()
Kaggle_cleaned = Kaggle_cleaned[Kaggle_cleaned['Query'].str.strip() != '']


# Разделение данных: запросы (X) и метки (y)
X = Kaggle_cleaned['Query'] # Тексты SQL-запросов
y = Kaggle_cleaned['Label'] # 0 - безопасный запрос, 1 - SQL-инъекция

# Разделяем данные на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизация TF-IDF
vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1,2), lowercase=True)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# создание модели
model = RandomForestClassifier(
    n_estimators = 200, # Используем 200 деревьев
    random_state =42, # Фиксирует случайность, чтобы при каждом запуске получалось одинаковое обучение.
    class_weight='balanced' # Автоматическая балансировка классов
 )

# Обучение модели
model.fit(X_train_vect, y_train)

# Сохранение модели
import pickle

with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Сохранение TF-IDF векторизатора
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Модель и векторизатор успешно сохранены")

# Предсказание
y_pred = model.predict(X_test_vect)

# Примеры SQL-запросов
test_queries = ["SELECT username FROM users WHERE id = 1", # безопасный
                "SELECT * FROM products ORDER BY price DESC", # безопасный
                "admin' --", # SQL-инъекция (комментарий)
                "SELECT * FROM users WHERE username = 'admin' OR '1'='1'", # SQL-инъекция (всегда истина)
                "DROP TABLE users", # SQL-инъекция (удаление таблицы)
                "INSERT INTO orders (id, product) VALUES (10, 'Book')", # безопасный
                "UNION SELECT null, username, password FROM users", # SQL-инъекция (UNION)
                "SELECT COUNT(*) FROM transactions WHERE amount > 100", # безопасный
                "SELECT * FROM accounts WHERE user_id IN (SELECT id FROM admins)", # безопасный
                "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)", # безопасный
                "admin' UNION SELECT password FROM users --", # SQL-инъекция (UNION с комментариями)
                "SELECT COUNT(*) FROM transactions WHERE amount > 100", # безопасный
                "SELECT * FROM users WHERE username = 'admin' OR '1'='1'", # SQL-инъекция
                "UPDATE users SET password='newpassword' WHERE username='admin'", # безопасный
                "' OR 1=1 --", # SQL-инъекция
                "' OR EXISTS (SELECT * FROM users WHERE username = 'admin') --", # SQL-инъекция
                "admin' --", # SQL-инъекция
                "SELECT * FROM customers WHERE email LIKE '%gmail.com'", # безопасный
                "'; EXEC sp_configure 'show advanced options', 1; RECONFIGURE; EXEC xp_cmdshell('whoami') --", # SQL-инъекция
                "SELECT username FROM users WHERE id = 1", # безопасный
                "' OR (SELECT COUNT(*) FROM users) > 0 --", # SQL-инъекция
                "SELECT department, COUNT(*) FROM employees GROUP BY department", # безопасный
                "INSERT INTO users (username, password) VALUES ('admin', ''); DROP TABLE users;", # SQL-инъекция
                "SELECT SUM(amount) FROM transactions WHERE type='purchase'", # безопасный
                "DELETE FROM transactions WHERE amount < 10;", # SQL-инъекция
                "SELECT * FROM products WHERE price >= 100;" # безопасный
]

# истинные метки для нового набора данных
true_labels_for_new_queries = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Векторизация
test_queries_vect = vectorizer.transform(test_queries)

# Предсказание
predictions = model.predict(test_queries_vect)

# Оценка точности модели
print(" Оценка по тестовым данным:")
print(classification_report(y_test, y_pred))

# Матрица ошибок для тестовых данных
print(" Матрица ошибок (тестовые данные):")
print(confusion_matrix(y_test, y_pred))

# Анализ неверных предсказаний для тетсовых данных
wrong_predictions_test = np.where(y_pred != y_test)[0] # np.where находит индексы ошибок (позиции, где y_pred ≠ y_test). [0] - извлекает массив индексов неверных предсказаний.
print(f" Количество неверных предсказаний на тестовых данных: {len(wrong_predictions_test)}") # len количество неверных предсказаний в массиве
print(" Ошибочные предсказания на тестовой выборке (первые 5):")

for i in wrong_predictions_test[:5]: # Вывод первых 5 ошибок
    print(f"Запрос: {X_test.iloc[i]} → Истинное значение: {y_test.iloc[i]}, Предсказание: {y_pred[i]}") # .iloc[i] — это метод, который позволяет получить данные по индексу

print(" Оценка по новым данным:")
print(classification_report(true_labels_for_new_queries, predictions))

# . Матрица ошибок для новых SQL-запросов
print(" Матрица ошибок (новые данные):")
print(confusion_matrix(true_labels_for_new_queries, predictions))

# Анализ неверных предсказаний для новых данных
wrong_predictions_new = np.where(predictions != true_labels_for_new_queries)[0]
print(f"\nКоличество ошибочных предсказаний на новых данных: {len(wrong_predictions_new)}")
print(" Ошибочные предсказания на новых данных (первые 5):")

for i in wrong_predictions_new[:5]:
    print(f"Запрос: {test_queries[i]} → Истинное значение: {true_labels_for_new_queries[i]}, Предсказание: {predictions[i]}")

feature_importances = np.argsort(model.feature_importances_)[::-1] # Сортируем важность признаков([::-1] - переворачивает список, чтобы получить самые важные слова первыми.)
feature_names = vectorizer.get_feature_names_out() # возвращает список слов и биграмм,

# Выводим ТОП-30 признаков
print("Топ 30 признаков, влияющих на предсказание:", feature_names[feature_importances[:30]])



