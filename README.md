# ML-система оцінки креативності / відповідності темі тексту

> Інтерпретована ML-система, що автоматично оцінює **креативність** або **релевантність темі** текстового контенту без використання LLM.  
> Підтримує **англійську та українську мови**

---

## Стек
**Python**, `pandas`, `numpy`, `scikit-learn`, `matplotlib`  
→ rule-based скоринг + `LinearCalibrator (Ridge)` для узгодження з людськими оцінками

---

## Структура
```
CREATIVITY_SCORE_NLP/
├── data/ # IELTS essays
├── src/ # модулі features / model / evaluation
└── notebook/ # main pipeline (creativity_scoring.ipynb)
```
---

## Основні ідеї
- **Фічі:** лексичне різноманіття, рідкісність, структурна складність, ентропія, topic similarity  
- **Rule-based оцінка:** прозора формула з вагами, шкала 0–100  
- **ML калібрування:** Ridge-регресія для точнішої відповідності людським оцінкам  
- **Двомовність:** токенізація та стоп-слова для 🇬🇧 і 🇺🇦

---

## Швидкий старт
```
pip install -r requirements.txt
jupyter notebook notebook/creativity_scoring.ipynb
```

## Результати
Метрика	Значення
MAE	0.30
Spearman ρ	0.145
Pearson r	0.106

**Найвпливовіші фічі: avg_word_len, hapax_ratio, bigram_entropy.**

## Висновок

Реалізовано всі вимоги ТЗ:

- прозора rule-based система,

- опціональне ML-калібрування,

- двомовна підтримка (ENG + UKR),

- узгодженість із людськими оцінками.


## 📚 Dataset: Raw IELTS Essays (Kaggle)


## 👩‍💻 Author: Veronika Pasichnyk