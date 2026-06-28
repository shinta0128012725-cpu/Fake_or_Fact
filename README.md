# 📰 Fake News Classification with Classical NLP & ML

英語ニュース記事をフェイクニュース（Fake News）と事実報道（Factual News）に分類する自然言語処理プロジェクト。  
深層学習に頼らず、**カウントベース・ルールベースの手法を意図的に採用**し、NLPの基礎をしっかり理解した上でモデルを構築することを目的としている。

---

## 🎯 プロジェクトの目的と技術選定の意図

現在のNLPタスクの多くは、BERTやGPTなどの事前学習済みモデルを転移学習・ファインチューニングして解くのが主流となっている。  
しかし本プロジェクトでは、**あえてカウントベース（Bag-of-Words, TF-IDF）やルールベース（VADER）の手法を中心に使用**している。

その理由は、深層学習モデルを使いこなすためにこそ、その「下の層」にある古典的NLP手法の動作原理をしっかり理解しておくことが重要だと考えているため。  
特徴量の設計・前処理・テキスト表現の選択が精度にどう影響するかを肌で理解することが、このプロジェクトの核心にある。



---

## 📊 EDA（探索的データ分析）

### データ概要

- **対象データ:** `fake_news_data.csv`
- **ターゲットラベル:** `fake_or_factual`（`Fake News` / `Factual News`）
- **分析対象カラム:** `text`（記事本文）

### 品詞タグ付けと固有表現抽出（spaCy）

spaCy（`en_core_web_sm`）を使用して各記事をトークン化し、以下を抽出・集計した。

- **品詞タグ（POS tag）:** NOUN（名詞）、VERB（動詞）などの品詞ごとの出現頻度
- **固有表現（NER tag）:** ORG（組織）、GPE（地名）、PERSON（人名）、NORP（国籍・宗教）など

Fake NewsとFactual Newsそれぞれで固有表現の分布を可視化したことで、以下のような傾向が確認できた。

- Fake Newsでは特定の**人名（PERSON）や組織名（ORG）**が偏って多く登場する
- Factual Newsでは**地名（GPE）や数値（CARDINAL）**の出現が相対的に多い

この分析により、フェイクニュースは特定の固有名詞に依存したナラティブを持ちやすいという仮説を立てることができた。

### N-gram分析（前処理後）

テキスト前処理後のトークンでUnigramおよびBigramの頻出度を集計し、Fake / Factual別に可視化した。

- Fake Newsに頻出するbigramには、特定の政治的文脈を持つ語の組み合わせが多く見られた
- Factual Newsでは、中立的・報道的な表現のbigramが上位を占める傾向があった

### 感情分析（VADER）

VADERによるsentiment scoreを算出し、Fake / Factual別に感情傾向を比較した。

- **分類基準:** score < -0.1 → Negative / -0.1 ≤ score ≤ 0.1 → Neutral / score > 0.1 → Positive
- Fake Newsはネガティブな文章の割合が高い傾向が見られた
- Factual Newsはニュートラル〜ポジティブな文章が多い傾向があった

VADERはルールベースの感情分析器であり、文脈理解は限定的だが、ニュース記事の感情傾向を素早くスコア化できる点で有効。このスコアはのちにXGBoostモデルの追加特徴量としても活用している。

### トピックモデリング（LDA・LSA + TF-IDF）

gensimを使い、Fake Newsのトピック構造を分析した。

- **LDA（Latent Dirichlet Allocation）:** coherence scoreを指標にトピック数を検証し、最適なトピック数8でモデルを構築
- **LSA（Latent Semantic Analysis）+ TF-IDF:** TF-IDF変換後のコーパスにLSAを適用し、意味的なトピック抽出を実施
- Fake NewsとFactual News両方でトピックの傾向を比較し、語彙・テーマの違いを確認した

---

## ⚙️ テキスト前処理パイプライン

```
原文テキスト
  └─ ハイフン・前置き文字列の除去（regex）
  └─ 小文字化
  └─ 記号・特殊文字の除去（regex）
  └─ ストップワード除去（NLTK）
  └─ トークン化（NLTK word_tokenize）
  └─ 見出し語化（WordNetLemmatizer）
```

---

## 🤖 分類モデル

### 特徴量

| 特徴量 | 手法 |
|--------|------|
| テキスト表現 | CountVectorizer（Bag-of-Words） |
| 感情スコア | VADER compound score |

### 使用モデルと結果比較

| モデル | 特徴量 | 備考 |
|--------|--------|------|
| Logistic Regression | BoW | 線形モデルのベースライン |
| Linear SVM（SGDClassifier） | BoW | 高次元テキストデータに強い線形モデル |
| XGBoost | BoW + VADERスコア | 感情スコアを追加特徴量として組み合わせ |

---

## 🛠️ 使用技術・ライブラリ

| カテゴリ | ライブラリ |
|----------|-----------|
| データ処理 | pandas |
| テキスト処理 | spaCy, NLTK |
| 感情分析 | VADER (vaderSentiment) |
| トピックモデリング | gensim (LDA, LSA, TF-IDF) |
| 特徴量抽出 | scikit-learn (CountVectorizer, TfidfVectorizer) |
| 分類モデル | scikit-learn (LogisticRegression, SGDClassifier), XGBoost |
| 可視化 | matplotlib, seaborn |

---

## 📁 ファイル構成

```
.
├── NLP_ML_フェイクニュース判別.ipynb   # メインノートブック
├── fake_news_data.csv                  # 使用データ
└── README.md
```
