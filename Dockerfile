# Используем базовый образ Python
FROM python:3.11.6

# Установка зависимостей Python
RUN pip install pandas scikit-learn catboost

# Копируем файлы в контейнер
COPY main.py /
COPY train_df.csv /
COPY test_df.csv /

# Команда, которая будет выполняться при запуске контейнера
CMD ["python", "main.py"]
