import json
import os
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Настройки путей ---
WORK_DIR = "./LABS/AI/lab1/2026_kidbook_153/WORK/2.1_society/how_and_where_find_friends"
WEB_BASE_DIR = "./LABS/AI/lab1/2026_kidbook_153/WEB"  # Базовая директория WEB
CONCEPTS_FILE = os.path.join(WORK_DIR, "concepts.json")
REPORT_FILE = os.path.join(WORK_DIR, "linking_suggestions.md")

# Параметры поиска
SIMILARITY_THRESHOLD = 0.08  # Порог схожести. Если ссылок мало, снизьте до 0.08
MAX_LINKS = 20                # Максимальное количество ссылок на одну статью

def get_all_md_files(base_directory):
    """
    Рекурсивно находит все .md файлы во всех подкаталогах WEB.
    """
    documents = []
    md_files = glob.glob(os.path.join(base_directory, "**/*.md"), recursive=True)
    
    print(f"Найдено .md файлов во всех подкаталогах WEB: {len(md_files)}")
    
    for full_path in md_files:
        try:
            # Получаем относительный путь от WEB_DIR для формирования ID
            rel_path = os.path.relpath(full_path, base_directory)
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Извлекаем заголовок из Markdown (первый заголовок h1)
                title = extract_title_from_md(content)
                if not title:
                    # Если нет заголовка, используем имя файла
                    title = os.path.basename(full_path).replace(".md", "").replace("_", " ").title()
                
                documents.append({
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "web_path": os.path.join("WEB", rel_path).replace("\\", "/"),  # для ID в формате WEB/...
                    "title": title,
                    "content": content,
                    "filename": os.path.basename(full_path)
                })
        except Exception as e:
            print(f"Ошибка при чтении {full_path}: {e}")
    
    return documents

def extract_title_from_md(content):
    """Извлекает первый заголовок h1 из Markdown."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def clean_text(text):
    """Очищает текст от Markdown разметки."""
    # Убираем ссылки [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Убираем изображения
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # Убираем форматирование
    text = re.sub(r'[*_`#]', '', text)
    # Приводим к нижнему регистру
    text = text.lower()
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text

def find_related_articles(concept_text, all_articles, vectorizer, article_vectors, top_n=MAX_LINKS):
    """
    Находит статьи, похожие на заданный текст концепта.
    """
    # Векторизуем текст концепта
    concept_vector = vectorizer.transform([concept_text])
    
    # Вычисляем схожесть со всеми статьями
    similarities = cosine_similarity(concept_vector, article_vectors).flatten()
    
    # Получаем индексы статей с схожестью выше порога
    similar_indices = np.where(similarities >= SIMILARITY_THRESHOLD)[0]
    
    # Сортируем по убыванию схожести
    similar_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
    
    # Формируем результаты
    results = []
    for idx in similar_indices[:top_n]:
        results.append({
            "article": all_articles[idx],
            "similarity": float(similarities[idx])
        })
    
    return results

def format_markdown_link(article):
    """Форматирует Markdown ссылку на статью."""
    return f"[{article['title']}]({article['web_path']})"

def main():
    print("=" * 60)
    print("Запуск анализа перелинковки концептов")
    print("=" * 60)
    
    # 1. Загружаем концепты (целевые леммы)
    if not os.path.exists(CONCEPTS_FILE):
        print(f"❌ Критическая ошибка: Файл {CONCEPTS_FILE} не найден.")
        return
    
    print(f"📂 Загрузка концептов из: {CONCEPTS_FILE}")
    with open(CONCEPTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Извлекаем плоский список концептов
    target_concepts = []
    for section in data:
        section_name = section.get("section", "Без раздела")
        for concept in section.get("concepts", []):
            # Собираем текст для анализа (название + леммы + описание)
            analysis_text = f"{concept.get('name', '')} {' '.join(concept.get('lemmas', []))} {concept.get('description', '')}"
            analysis_text = clean_text(analysis_text)
            
            target_concepts.append({
                "id": concept["id"],
                "name": concept["name"],
                "file": concept["file"],
                "lemmas": concept.get("lemmas", []),
                "description": concept.get("description", ""),
                "analysis_text": analysis_text,
                "section": section_name,
                "author": concept.get("author", "Неизвестен")
            })
    
    print(f"✅ Загружено концептов: {len(target_concepts)}")

    # 2. Загружаем все статьи из ВСЕХ подкаталогов WEB
    print(f"📂 Рекурсивный поиск .md файлов в: {WEB_BASE_DIR}")
    web_articles = get_all_md_files(WEB_BASE_DIR)
    
    if not web_articles:
        print("❌ Статьи не найдены в WEB директории.")
        return
    
    if not target_concepts:
        print("❌ Концепты не найдены.")
        return
    
    print(f"✅ Загружено статей из всех подкаталогов: {len(web_articles)}")

    # 3. Подготавливаем тексты статей для векторизации
    print("📊 Подготовка текстов для векторизации...")
    article_clean_texts = [clean_text(a["content"]) for a in web_articles]

    # 4. Векторизация через TF-IDF
    print("🔄 Вычисление TF-IDF векторов...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Ограничиваем количество признаков
        min_df=2,           # Игнорируем слова, встречающиеся менее чем в 2 документах
        max_df=0.8,         # Игнорируем слишком частые слова
        ngram_range=(1, 2),  # Используем униграммы и биграммы
        stop_words=['это', 'как', 'что', 'для', 'в', 'на', 'с', 'по', 'и', 'не', 'весь', 'мочь', 'быть']
    )
    
    # Векторизуем статьи
    article_vectors = vectorizer.fit_transform(article_clean_texts)
    print(f"   Размерность матрицы: {article_vectors.shape}")

    # 5. Находим связи для каждого концепта
    print("🔗 Поиск связей между концептами и статьями...")
    
    all_suggestions = []
    
    for i, concept in enumerate(target_concepts):
        if i % 10 == 0 and i > 0:
            print(f"   Обработано {i}/{len(target_concepts)} концептов")
        
        # Ищем похожие статьи
        related = find_related_articles(
            concept["analysis_text"], 
            web_articles, 
            vectorizer, 
            article_vectors,
            top_n=MAX_LINKS
        )
        
        if related:
            all_suggestions.append({
                "concept": concept,
                "related_articles": related
            })

    print(f"✅ Найдено связей для {len(all_suggestions)} концептов")

    # 6. Генерируем MD отчет
    print("📝 Генерация отчета...")
    with open(REPORT_FILE, 'w', encoding='utf-8') as md:
        md.write("# Отчет по перелинковке концептов\n\n")
        md.write(f"*Дата генерации: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        md.write("## 📊 Статистика\n\n")
        md.write(f"- **Всего концептов:** {len(target_concepts)}\n")
        md.write(f"- **Всего статей в WEB:** {len(web_articles)}\n")
        md.write(f"- **Концептов со связями:** {len(all_suggestions)}\n")
        md.write(f"- **Порог схожести:** {SIMILARITY_THRESHOLD}\n\n")
        
        md.write("## 🔗 Предлагаемые связи\n\n")
        
        for item in all_suggestions:
            concept = item["concept"]
            
            md.write(f"### 📄 {concept['name']}\n\n")
            md.write(f"- **ID:** `{concept['id']}`\n")
            md.write(f"- **Файл:** `{concept['file']}`\n")
            md.write(f"- **Раздел:** {concept['section']}\n")
            md.write(f"- **Автор:** {concept['author']}\n\n")
            
            if concept['lemmas']:
                md.write(f"**Леммы:** {', '.join(concept['lemmas'][:10])}\n\n")
            
            md.write("#### 🔍 Рекомендуемые связанные статьи:\n\n")
            
            for rel in item["related_articles"]:
                article = rel["article"]
                similarity = rel["similarity"]
                
                # Визуальный индикатор схожести
                if similarity > 0.3:
                    strength = "🔴 Сильная"
                elif similarity > 0.2:
                    strength = "🟡 Средняя"
                else:
                    strength = "🟢 Слабая"
                
                md.write(f"1. **{article['title']}**\n")
                md.write(f"   - Путь: `{article['web_path']}`\n")
                md.write(f"   - Схожесть: {similarity:.3f} ({strength})\n")
                md.write(f"   - Ссылка: [{article['title']}](../../../{article['web_path']})\n\n")
            
            # Генерируем HTML-блок для вставки в конец статьи
            md.write("#### 📋 HTML для вставки в конец статьи:\n\n")
            md.write("```html\n")
            md.write('<div class="related-articles">\n')
            md.write('  <h3>Читайте также</h3>\n')
            md.write('  <ul>\n')
            
            for rel in item["related_articles"]:
                article = rel["article"]
                md.write(f'    <li><a href="{article["web_path"]}">{article["title"]}</a></li>\n')
            
            md.write('  </ul>\n')
            md.write('</div>\n')
            md.write("```\n\n")
            
            # Генерируем Markdown для вставки
            md.write("#### 📝 Markdown для вставки:\n\n")
            md.write("```markdown\n")
            md.write("## Читайте также\n\n")
            
            for rel in item["related_articles"]:
                article = rel["article"]
                md.write(f"- [{article['title']}]({article['web_path']})\n")
            
            md.write("```\n\n")
            md.write("---\n\n")
        
        # Добавляем раздел со всеми статьями для справки
        md.write("## 📚 Все доступные статьи в WEB\n\n")
        for article in web_articles:
            md.write(f"- [{article['title']}]({article['web_path']})\n")

    print(f"✅ Отчет сохранен в: {REPORT_FILE}")
    
    # Выводим краткую статистику
    print("\n" + "=" * 60)
    print("КРАТКАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего концептов: {len(target_concepts)}")
    print(f"Всего статей: {len(web_articles)}")
    print(f"Найдено связей: {sum(len(item['related_articles']) for item in all_suggestions)}")
    print(f"Среднее связей на концепт: {sum(len(item['related_articles']) for item in all_suggestions) / len(all_suggestions) if all_suggestions else 0:.2f}")
    print(f"Отчет сохранен: {REPORT_FILE}")

if __name__ == "__main__":
    main()