\# Assistant

A voice assistant built for educational practice (Chapter 5.3). The project includes a set of baseline features and is designed to be extended with new functionality by the student.

---

## Included Features

| Feature | Description |
|---------|-------------|
| Random greeting / farewell | Picks a random phrase each time the assistant starts or stops |
| Weather lookup | Reports current weather for a specified city |
| Translation | Translates text from a studied language into the user's native language |
| Web search | Performs a search query and returns results |
| Social media lookup | Searches for a person across social networks by name |

---

## Task

1. Read the source code carefully and understand how each function works.
2. If something is unclear, consult an AI assistant (e.g. DeepSeek) for a detailed explanation.
3. Add new features you consider useful — you are not limited to practical tasks.
4. The assistant may have a personality: a sense of humor, a specific tone, or any other character you choose.

---

## Getting Started

Read the detailed setup guide in `Instruction manual.txt` located in this repository.

---

## Project Structure

```
assistant/
├── main.py                  # Entry point
├── Instruction manual.txt    # Setup guide (read this first)
├── src/
│   ├── greeting.py           # Random greeting / farewell
│   ├── weather.py            # Weather lookup
│   ├── translator.py         # Translation module
│   ├── search.py             # Web search
│   └── social_search.py      # Social media person lookup
├── data/                     # Phrase banks, configs
└── README.md
```

---

## Suggested Extensions

- Voice synthesis and recognition (TTS / STT)
- Reminders and scheduled tasks
- Currency or unit conversion
- Joke of the day
- Music playback control
- Integration with a calendar or to-do list
- Custom personality module (formal, humorous, sarcastic, etc.)

---

## License

MIT

---

# Assistant

Голосовой ассистент, разработанный для учебной практики (глава 5.3). Проект включает набор базовых функций и предназначен для расширения новыми возможностями.

---

## Встроенные функции

| Функция | Описание |
|---------|----------|
| Случайное приветствие / прощание | Выбирает случайную фразу при запуске или завершении работы |
| Определение погоды | Сообщает текущую погоду для указанного города |
| Перевод | Перевод текста с изучаемого языка на родной язык пользователя |
| Поисковой запрос | Выполняет поиск в интернете и возвращает результаты |
| Поиск человека | Ищет человека в социальных сетях по имени |

---

## Задача

1. Внимательно изучите исходный код и разберитесь, как работает каждая функция.
2. Если что-то непонятно, обратитесь к нейросети (например, DeepSeek) за пояснением.
3. Добавьте новые функции, которые считаете нужными, — не ограничивайтесь практичными задачами.
4. Ассистент может иметь характер: чувство юмора, определённый тон или любой другой стиль общения.

---

## Запуск

Перед запуском ознакомьтесь с подробной инструкцией в файле `Instruction manual.txt`, который находится в данном репозитории.

---

## Структура проекта

```
assistant/
├── main.py                  # Точка входа
├── Instruction manual.txt    # Инструкция (прочитать в первую очередь)
├── src/
│   ├── greeting.py           # Случайное приветствие / прощание
│   ├── weather.py            # Определение погоды
│   ├── translator.py        # Модуль перевода
│   ├── search.py             # Поиск в интернете
│   └── social_search.py      # Поиск человека в соцсетях
├── data/                     # Банки фраз, конфиги
└── README.md
```

---

## Возможные расширения

- Синтез и распознавание речи (TTS / STT)
- Напоминания и планировщик задач
- Конвертер валют или единиц измерения
- Шутка дня
- Управление воспроизведением музыки
- Интеграция с календарём или списком дел
- Модуль личности (формальный, юмористический, саркастический и т. д.)

---

## Лицензия

MIT
