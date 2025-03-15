# 필요한 라이브러리 임포트
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import language_v1
from transformers import pipeline
import csv
from sentistrength import PySentiStr
import nltk
from nltk.corpus import wordnet
import feedparser
import time
import re
from random import randint
import seaborn as sns  # 추가
import numpy as np
from scipy.stats import zscore
from wordcloud import WordCloud  # 워드클라우드 라이브러리 추가
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# 코드 시작 부분에 추가
load_dotenv()  # .env 파일 로드

# Google API 인증 파일 경로 (실제 경로로 업데이트 필요)
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# 필요한 NLTK 데이터 다운로드
try:
    nltk.download("wordnet", quiet=True)
except:
    print("NLTK wordnet 다운로드 실패 - 오프라인이거나 NLTK 설치에 문제가 있습니다.")

# 뉴스 소스 정의
NEWS_SOURCES = {
    "USA": {
        "The New York Times": {"bias": "liberal"},
        "CNN": {"bias": "liberal"},
        "Fox News": {"bias": "conservative"}
    },
    "UK": {
        "The Guardian": {"bias": "liberal"},
        "The Telegraph": {"bias": "conservative"},
        "BBC News": {"bias": "neutral"}
    }
}

# 추가할 뉴스 소스 업데이트 (main() 실행 전에 위치시킴)
NEWS_SOURCES["USA"].update({
    "The Washington Post": {"bias": "liberal"},
    "The Wall Street Journal": {"bias": "conservative"},
    "Bloomberg": {"bias": "neutral"},
    "Reuters": {"bias": "neutral"}
})

NEWS_SOURCES["UK"].update({
    "The Times": {"bias": "conservative"},
    "The Independent": {"bias": "liberal"},
    "Financial Times": {"bias": "neutral"},
    "Daily Mail": {"bias": "conservative"}
})

# 주제별 키워드 정의
TOPIC_KEYWORDS = {
    "Politics": ["election fraud", "immigration policy", "climate policy", "gun control", 
                 "voter suppression", "political polarization", "presidential debate", "democracy crisis"],
    "Economy": ["interest rate hike", "inflation crisis", "recession fears", "stock market crash", 
                "crypto regulation", "economic inequality", "unemployment rate", "consumer confidence index"],
    "Society": ["LGBTQ rights", "racial discrimination", "mental health crisis", "police brutality", 
                "cancel culture", "misinformation", "online harassment", "censorship"],
    "International Relations": ["Ukraine Russia war", "China Taiwan tension", "Middle East crisis", "UN sanctions", 
                                "trade war", "diplomatic conflict", "international peace talks", "NATO expansion"]
}

def analyze_topic_sentiments(df):
    """주제별 감정 점수와 감정 강도를 계산"""
    topic_results = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        topic_df = df[df['search_keyword'].isin(keywords)]
        
        if topic_df.empty:
            continue
        
        topic_sentiment_avg = topic_df["final_sentiment_score"].mean()
        topic_intensity_avg = topic_df["sentiment_intensity_score"].mean()
        
        topic_results.append({
            "topic": topic,
            "avg_sentiment_score": topic_sentiment_avg,
            "avg_sentiment_intensity": topic_intensity_avg
        })

    return pd.DataFrame(topic_results)

# 결과 저장 경로 지정
RESULTS_DIR = "/home/hwangjeongmun691/언론별 감정 분석/결과"

# 디버그 모드 설정
DEBUG_MODE = True

def debug_print(message, data=None, important=False):
    """디버그 메시지 출력 함수"""
    if DEBUG_MODE:
        if important:
            print("\n" + "="*50)
            print(f"🔍 {message}")
            print("="*50)
        else:
            print(f"🔹 {message}")
        
        if data is not None:
            if isinstance(data, str) and len(data) > 300:
                print(f"{data[:300]}... (생략됨)")
            else:
                print(data)

# 오류 메시지 설정
def setup_error_handling():
    """오류 처리 설정"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("알림: 국가별, 신문사별 감정 강도 비교를 위한 분석을 시작합니다.")

# WordNet을 활용한 동의어 확장
def get_synonyms(keyword):
    """단어의 동의어 목록 반환"""
    synonyms = set()
    try:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
    except Exception as e:
        debug_print(f"동의어 찾기 오류: {e}")
    return list(synonyms)

# 소스 도메인 가져오기
def get_source_domain(source):
    """뉴스 소스의 도메인 반환"""
    source_domains = {
        'CNN': 'cnn.com',
        'Fox News': 'foxnews.com',
        'The Guardian': 'theguardian.com',
        'The New York Times': 'nytimes.com',
        'BBC News': 'bbc.com,bbc.co.uk',
        'The Telegraph': 'telegraph.co.uk',
        'Reuters': 'reuters.com',
        'CNBC': 'cnbc.com',
        'Bloomberg': 'bloomberg.com'
    }
    
    return source_domains.get(source, '')

# 소스 일치 여부 확인
def is_same_source(found_source, target_source):
    """소스 이름을 유연하게 비교하여 일치 여부를 반환"""
    found_lower = found_source.lower().strip()
    target_lower = target_source.lower().strip()

    # 완전히 일치하는 경우
    if found_lower == target_lower:
        return True
    
    # 부분 문자열 포함 여부 확인 (CNN ↔ CNN International)
    if target_lower in found_lower or found_lower in target_lower:
        return True
    
    # 도메인 기반 소스 확인
    source_domains = {
        "CNN": "cnn.com",
        "Fox News": "foxnews.com",
        "The Guardian": "theguardian.com",
        "The New York Times": "nytimes.com",
        "BBC News": "bbc.com",
        "The Telegraph": "telegraph.co.uk",
        "Reuters": "reuters.com",
        "Bloomberg": "bloomberg.com",
        "The Times": "thetimes.co.uk",
        "The Independent": "independent.co.uk",
        "Financial Times": "ft.com",
        "Daily Mail": "dailymail.co.uk",
        "The Wall Street Journal": "wsj.com",
        "The Washington Post": "washingtonpost.com"
    }

    if target_source in source_domains and source_domains[target_source] in found_lower:
            return True
    
    # 특수 케이스 처리
    special_cases = {
        "the times": ["the times", "the times uk", "the sunday times"],
        "the new york times": ["nyt", "ny times", "new york times"],
    }
    
    if target_lower in special_cases:
        target_variants = special_cases[target_lower]
        if found_lower in target_variants:
            return True
        # 특수 케이스 예외 처리
        if target_lower == "the times" and "new york" in found_lower:
            return False
        if target_lower == "the new york times" and found_lower == "the times":
            return False

    return False

# Google News RSS에서 뉴스 검색
def get_google_news_rss(keyword, source=None):
    """Google News RSS에서 키워드로 뉴스 검색 (소스 매칭 개선)"""
    query = keyword
    if source:
        query = f"{keyword} site:{get_source_domain(source)}"
    
    query = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    debug_print(f"Google News RSS 요청 URL: {url}")
    
    try:
        feed = feedparser.parse(url)
        articles = []

        for entry in feed.entries[:15]:  # 기사 15개 가져오기
            title_parts = entry.title.split(" - ")
            entry_source = title_parts[-1].strip() if len(title_parts) > 1 else "Unknown"
            article_url = entry.link

            # 소스 이름 표준화
            normalized_source = normalize_source_name(entry_source)
            
            # 소스 확인 - 더 엄격한 검사
            if source and not is_same_source(normalized_source, source):
                debug_print(f"소스 불일치: '{normalized_source}' ≠ '{source}' - 건너뜀")
                continue
                
            article = {
                "title": title_parts[0].strip(),
                "content": entry.description if hasattr(entry, "description") else title_parts[0].strip(),
                "source": normalized_source,  # 표준화된 소스 이름 사용
                "url": article_url,
                "published_at": entry.published if hasattr(entry, "published") else datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            articles.append(article)
        
        debug_print(f"'{keyword}' 검색 결과: {len(articles)}개 기사 발견")
        return articles
        
    except Exception as e:
        debug_print(f"Google News RSS 요청 중 오류 발생: {str(e)}")
        return []

# 키워드 확장 + Google News 검색 결합
def search_expanded_news(keyword, source=None):
    """키워드 확장 후 뉴스 검색"""
    debug_print(f"'{keyword}' 확장 검색 시작 (소스: {source})", important=True)
    
    # Step 1: 원래 키워드로 Google News 검색
    initial_news = get_google_news_rss(keyword, source)
    
    if len(initial_news) >= 5:  # 충분한 결과가 있으면 바로 반환
        debug_print(f"원래 키워드로 충분한 결과({len(initial_news)}개)를 찾았습니다.")
        return initial_news[:5]  # 상위 5개만 반환
    
    # Step 2: 키워드 확장이 필요한 경우
    debug_print(f"결과가 부족하여 키워드 확장을 시도합니다.")
    expanded_keywords = [keyword]  # 원래 키워드 포함
    
    # 단일 단어인 경우 WordNet 동의어 추가
    if ' ' not in keyword and len(keyword) > 3:
        synonyms = get_synonyms(keyword)[:3]  # 상위 3개 동의어만 사용
        expanded_keywords.extend(synonyms)
    
    # 중복 제거
    expanded_keywords = list(set(expanded_keywords))
    debug_print(f"확장된 키워드: {expanded_keywords}")
    
    # Step 3: 확장된 키워드로 추가 검색
    all_news = initial_news.copy()  # 초기 결과 포함
    
    for k in expanded_keywords:
        if k == keyword:  # 원래 키워드는 이미 검색했으므로 건너뜀
            continue
            
        debug_print(f"확장 키워드 '{k}'로 검색 중...")
        additional_news = get_google_news_rss(k, source)
        
        # 중복 제거하며 추가
        for article in additional_news:
            # URL로 중복 확인
            if not any(existing['url'] == article['url'] for existing in all_news):
                all_news.append(article)
    
    debug_print(f"최종 검색 결과: {len(all_news)}개 기사")
    return all_news[:15]  # 더 많은 기사 반환

# 감정 분석 함수들

def get_vader_sentiment(text):
    """VADER를 사용하여 감정 점수 분석"""
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        # compound 점수는 -1에서 1 사이의 값
        compound_score = sentiment_scores['compound']
        print(f"VADER 원본 점수: {compound_score}")
        return compound_score
    except Exception as e:
        print(f"VADER 감정 분석 오류: {e}")
        return 0

def get_sentistrength_sentiment(text):
    """SentiStrength를 사용하여 감정 점수 분석"""
    try:
        from sentistrength import PySentiStr
        senti = PySentiStr()
        senti.setSentiStrengthPath("/home/hwangjeongmun691/언론별 감정 분석/SentiStrength.jar")
        senti.setSentiStrengthLanguageFolderPath("/home/hwangjeongmun691/언론별 감정 분석/SentiStrength_Data/")
        
        # dual 방식으로 긍정, 부정 점수 모두 받기
        result = senti.getSentiment(text, score='dual')
        if isinstance(result, list):
            result = result[0]  # 첫 번째 결과만 사용
        
        # 결과는 (긍정점수, 부정점수) 형태의 튜플
        return result[0], result[1]
    except Exception as e:
        print(f"SentiStrength 감정 분석 오류: {e}")
        return 1, -1  # 중립 값 반환

def get_google_sentiment(text):
    """Google Natural Language API를 사용하여 감정 점수 분석"""
    try:
        # 환경 변수 설정
        cred_path = "/home/hwangjeongmun691/언론별 감정 분석/comparative-sentiment-analysis-c0b363950560.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        
        # 인증 파일 확인
        if not os.path.exists(cred_path):
            print(f"⚠️ 인증 파일을 찾을 수 없습니다: {cred_path}")
            return 0
        
        # 클라이언트 초기화
        client = language_v1.LanguageServiceClient()
        
        # 문서 객체 생성
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        # 감정 분석 수행
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
        
        # Google API는 -1.0 ~ 1.0 범위의 점수를 반환
        score = sentiment.score
        print(f"Google API 원본 점수: {score}")
        return score
    except Exception as e:
        print(f"Google 감정 분석 오류: {e}")
        return 0  # 오류 발생 시 중립값 반환

def get_huggingface_sentiment(text):
    try:
        # 모델 명시적 지정으로 경고 제거
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english")
        # 텍스트 길이 제한 (너무 긴 텍스트는 잘라서 분석)
        max_length = 512
        truncated_text = text[:max_length] if len(text) > max_length else text
        result = sentiment_analyzer(truncated_text)[0]
        
        # NEGATIVE 결과에 대한 처리 개선
        if result['label'] == 'POSITIVE':
            score = result['score']
        else:
            score = 1 - result['score']  # 더 직관적인 변환
            
        print(f"Hugging Face 점수: {score} (원래 레이블: {result['label']})")
        return score
    except Exception as e:
        print(f"Hugging Face 감정 분석 오류: {e}")
        return 0.5

# 점수 정규화 함수들

def normalize_vader_score(score):
    """VADER 점수를 0~1 범위로 정규화"""
    normalized_score = (score + 1) / 2
    return normalized_score

def normalize_sentistrength_score(positive_score, negative_score):
    """SentiStrength 점수를 0~1 범위로 정규화"""
    combined_score = (positive_score + (6 + negative_score)) / 10
    return combined_score

def normalize_google_score(score):
    """Google API 점수를 0~1 범위로 정규화"""
    normalized_score = (score + 1) / 2
    return normalized_score

# 최종 감정 점수 계산 함수

def calculate_final_sentiment(vader_score, google_score, huggingface_score, df=None):
    """각 도구의 정규화된 감정 점수를 가중 평균하여 최종 감정 점수 계산 (극단값 조정 포함)"""
    
    # 기본 가중치 설정
    vader_weight = 0.4
    google_weight = 0.4
    huggingface_weight = 0.2

    # Z-score를 활용하여 감정 분석 도구의 점수 극단성 판단
    if df is not None:
        df = calculate_z_scores(df)  # Z-score 계산
        huggingface_zscore = df["huggingface_score_zscore"].iloc[-1]  # 마지막 데이터 기준
        vader_zscore = df["vader_score_zscore"].iloc[-1]
        google_zscore = df["google_score_zscore"].iloc[-1]

        # Hugging Face 감정 점수의 극단성 확인
        if abs(huggingface_zscore) > 2.0:
            huggingface_weight = 0.1
            print(f"Hugging Face 감정 점수 조정 (Z-score: {huggingface_zscore:.2f}) → 가중치 {huggingface_weight}")

        # VADER 감정 점수의 극단성 확인
        if abs(vader_zscore) > 2.0:
            vader_weight = 0.3
            print(f"VADER 감정 점수 조정 (Z-score: {vader_zscore:.2f}) → 가중치 {vader_weight}")

        # Google NLP 감정 점수의 극단성 확인
        if abs(google_zscore) > 2.0:
            google_weight = 0.3
            print(f"Google 감정 점수 조정 (Z-score: {google_zscore:.2f}) → 가중치 {google_weight}")

    # 정규화된 감정 점수 계산
    normalized_vader = normalize_vader_score(vader_score)
    normalized_google = normalize_google_score(google_score)

    # 최종 감정 점수 계산
    final_score = (vader_weight * normalized_vader + 
                   google_weight * normalized_google + 
                   huggingface_weight * huggingface_score)

    return final_score

# 기사 감정 분석 함수

def calculate_z_scores(df):
    """감정 점수 및 감정 강도를 Z-score로 변환"""
    z_score_df = df.copy()
    
    # 감정 점수 Z-score 적용
    z_score_df["final_sentiment_zscore"] = zscore(df["final_sentiment_score"])
    z_score_df["sentiment_intensity_zscore"] = zscore(df["sentiment_intensity_score"])
    
    # 감정 분석 도구별 Z-score 적용
    tools = ["vader_score", "google_score", "huggingface_score"]
    for tool in tools:
        z_score_df[f"{tool}_zscore"] = zscore(df[tool])

    return z_score_df

def analyze_article_sentiment(article, country, search_keyword=None, df=None):
    """기사 감정 분석 수행 및 Hugging Face 점수 조정 포함"""
    title = article["title"]
    content = article["content"]
    source = article["source"]
    text = f"{title}. {content}"

    print(f"\n입력 뉴스 기사: '{title}'")
    print(f"출처: {source} (국가: {country})")
    print(f"내용 일부: {content[:100]}...\n")

    # 감정 분석 도구 적용
    vader_score = get_vader_sentiment(text)
    google_score = get_google_sentiment(text)
    huggingface_score = get_huggingface_sentiment(text)
    sentistrength_pos, sentistrength_neg = get_sentistrength_sentiment(text)

    # 감정 점수 및 감정 강도 계산 (Hugging Face 가중치 조정 포함)
    final_sentiment_score = calculate_final_sentiment(vader_score, google_score, huggingface_score, df)
    sentiment_intensity_score = normalize_sentistrength_score(sentistrength_pos, sentistrength_neg)
    
    # 결과 출력
    print("\n===== 감정 분석 결과 =====")
    print(f"VADER 감정 점수: {vader_score} (정규화: {normalize_vader_score(vader_score):.4f})")
    print(f"Google API 감정 점수: {google_score} (정규화: {normalize_google_score(google_score):.4f})")
    print(f"Hugging Face 감정 점수: {huggingface_score:.4f}")
    print(f"최종 감정 점수: {final_sentiment_score:.4f}")
    print(f"SentiStrength 감정 강도: 긍정={sentistrength_pos}, 부정={sentistrength_neg} (정규화: {sentiment_intensity_score:.4f})")

    return {
        "title": title,
        "source": source,
        "country": country,
        "bias": get_source_bias(source),
        "search_keyword": search_keyword,
        "final_sentiment_score": final_sentiment_score,
        "sentiment_intensity_score": sentiment_intensity_score,
        "vader_score": normalize_vader_score(vader_score),
        "google_score": normalize_google_score(google_score),
        "huggingface_score": huggingface_score,
        "url": article.get("url", ""),
        "published_at": article.get("published_at", datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
    }

def get_source_bias(source):
    """신문사의 정치적 성향 반환"""
    for country, sources in NEWS_SOURCES.items():
        if source in sources:
            return sources[source]['bias']
    return "unknown"

# 국가별, 신문사별 감정 강도 비교 함수

def compare_news_sources(keywords, articles_per_source=3):
    """여러 뉴스 소스에서 키워드별 감정 분석 비교"""
    results = []
    
    # 원하는 뉴스 소스로 정확히 제한
    sources = {
        'CNN': 'US',
        'Fox News': 'US',
        'The Guardian': 'UK',
        'The New York Times': 'US',
        'The Telegraph': 'UK',
        'BBC News': 'UK'
    }
    
    for keyword in keywords:
        debug_print(f"\n키워드 '{keyword}'에 대한 분석 시작...", important=True)
        
        for source, country in sources.items():
            debug_print(f"{source}({country})에서 '{keyword}' 검색 중...")
            
            # Google News RSS로 기사 검색
            articles = search_expanded_news(keyword, source)
            
            if not articles:
                debug_print(f"❌ {source}에서 '{keyword}'에 대한 기사를 찾을 수 없습니다.")
                continue
            
            # 찾은 기사 개수 제한
            articles = articles[:articles_per_source]
            
            # 각 기사 분석
            for article in articles:
                # 중복 API 호출 방지를 위한 지연
                time.sleep(randint(1, 3))
                
                result = analyze_article_sentiment(article, country, search_keyword=keyword)
                
                if result:
                    results.append(result)
                    debug_print(f"✅ 분석 완료: {result['title']} (점수: {result['final_sentiment_score']})")
    
    return results

# 결과 시각화 및 저장 함수

def get_date_folder():
    """오늘 날짜를 기준으로 폴더명 생성 (예: 20240601)"""
    today = datetime.datetime.now()
    return today.strftime("%Y%m%d")

def get_timestamp():
    """현재 시간을 파일명에 적합한 형식으로 반환 (예: 20240601_143045)"""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def get_result_folder():
    """날짜별 결과 폴더 경로 반환 및 생성"""
    date_folder = get_date_folder()
    result_path = os.path.join(RESULTS_DIR, date_folder)
    
    # 결과 폴더가 없으면 생성
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # 날짜별 폴더가 없으면 생성
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"새 날짜 폴더 생성: {result_path}")
    
    return result_path

def sample_article_review(df, sample_size=5):
    """Sample article review for sentiment analysis validation"""
    print("\n🔍 Sample Article Sentiment Analysis Review")
    if len(df) < sample_size:
        sample_size = len(df)
        print(f"⚠️ Only {sample_size} samples available for review.")
    
    sample_articles = df.sample(sample_size)
    for idx, row in sample_articles.iterrows():
        print(f"\nArticle Title: {row['title']}")
        print(f"Source: {row['source']} (Country: {row['country']})")
        print(f"Search Keyword: {row.get('search_keyword', 'No info')}")
        
        # sentiment 컬럼이 없으므로 점수만 표시하거나 점수를 기준으로 감정 상태를 계산
        sentiment_label = "긍정적" if row['final_sentiment_score'] > 0.6 else "중립적" if row['final_sentiment_score'] > 0.4 else "부정적"
        print(f"Final Sentiment Score: {row['final_sentiment_score']:.4f} ({sentiment_label})")
        
        # SentiStrength가 없으면 출력에서 제외
        tool_scores = f"VADER={row['vader_score']:.2f}, Google={row['google_score']:.2f}, HuggingFace={row['huggingface_score']:.2f}"
        if 'sentistrength_score' in row:
            tool_scores = f"VADER={row['vader_score']:.2f}, SentiStrength={row['sentistrength_score']:.2f}, Google={row['google_score']:.2f}, HuggingFace={row['huggingface_score']:.2f}"
        
        print(f"Tool Scores: {tool_scores}")
        print(f"URL: {row.get('url', 'No info')}")
        print("=" * 50)

def visualize_sentiment_by_source(df, result_folder, timestamp):
    """신문사별 감정 점수 분포 시각화"""
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='source', y='final_sentiment_score', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title('Sentiment Score Distribution by News Source')
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'source_sentiment_boxplot_{timestamp}.png'))
    print(f"📊 신문사별 감정 점수 박스플롯 저장 완료.")

def compare_sentiment_tools(df, result_folder, timestamp):
    """Compare sentiment scores across different analysis tools"""
    try:
        # 사용 가능한 도구 확인
        available_tools = []
        tool_labels = []
        
        # 항상 존재하는 기본 도구들
        if 'vader_score' in df.columns:
            available_tools.append('vader_score')
            tool_labels.append('VADER')
            
        if 'google_score' in df.columns:
            available_tools.append('google_score')
            tool_labels.append('Google NLP')
            
        if 'huggingface_score' in df.columns:
            available_tools.append('huggingface_score')
            tool_labels.append('HuggingFace')
        
        # SentiStrength는 sentiment_intensity_score로 저장됨
        if 'sentiment_intensity_score' in df.columns:
            available_tools.append('sentiment_intensity_score')
            tool_labels.append('SentiStrength')
            
        if len(available_tools) < 2:
            print("⚠️ 도구 비교를 위한 충분한 데이터가 없습니다.")
            return
            
        # Box plot
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df[available_tools])
        ax.set_xticklabels(tool_labels)
        plt.title('Sentiment Score Comparison Across Analysis Tools')
        plt.ylabel('Sentiment Score (0-1)')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'tool_comparison_{timestamp}.png'))
        
        # Correlation analysis
        plt.figure(figsize=(10, 8))
        correlation = df[available_tools].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    xticklabels=tool_labels, yticklabels=tool_labels)
        plt.title('Correlation Between Sentiment Analysis Tools')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'tool_correlation_{timestamp}.png'))
        print(f"📊 감정 분석 도구 비교 그래프 저장 완료.")
    except Exception as e:
        print(f"⚠️ 도구 비교 중 오류 발생: {e}")

def save_topic_statistics(df, result_folder, timestamp):
    """주제별 감정 점수 및 감정 강도 통계를 CSV로 저장"""
    topic_stats = []

    # 전체 데이터 통계 계산
    overall_stats = {
        "전체 감정 점수 평균": df["final_sentiment_score"].mean(),
        "전체 감정 점수 표준편차": df["final_sentiment_score"].std(),
        "전체 감정 점수 분산": df["final_sentiment_score"].var(),
        "전체 감정 강도 평균": df["sentiment_intensity_score"].mean(),
        "전체 감정 강도 표준편차": df["sentiment_intensity_score"].std(),
        "전체 감정 강도 분산": df["sentiment_intensity_score"].var()
    }

    for topic, keywords in TOPIC_KEYWORDS.items():
        topic_df = df[df['search_keyword'].isin(keywords)]
        
        if topic_df.empty:
            continue
        
        topic_sentiment_avg = topic_df["final_sentiment_score"].mean()
        topic_sentiment_std = topic_df["final_sentiment_score"].std()
        topic_intensity_avg = topic_df["sentiment_intensity_score"].mean()
        topic_intensity_std = topic_df["sentiment_intensity_score"].std()
        
        topic_stats.append({
            "topic": topic,
            "avg_sentiment_score": topic_sentiment_avg,
            "std_sentiment_score": topic_sentiment_std,
            "avg_sentiment_intensity": topic_intensity_avg,
            "std_sentiment_intensity": topic_intensity_std
        })

    # 데이터프레임으로 변환
    topic_stats_df = pd.DataFrame(topic_stats)

    # CSV로 저장
    csv_path = f"{result_folder}/topic_statistics_{timestamp}.csv"
    topic_stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 전체 통계도 함께 저장
    with open(f"{result_folder}/overall_statistics_{timestamp}.csv", "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["항목", "값"])
        for key, value in overall_stats.items():
            writer.writerow([key, value])

    print(f"📊 주제별 감정 분석 통계 저장 완료: {csv_path}")
    print(f"📊 전체 감정 분석 통계 저장 완료: {result_folder}/overall_statistics_{timestamp}.csv")

def plot_topic_sentiments(df, result_folder, timestamp):
    """주제별 감정 점수와 감정 강도를 시각화"""
    topic_df = analyze_topic_sentiments(df)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=topic_df, x="topic", y="avg_sentiment_score", palette="coolwarm")
    plt.title("Average Sentiment Score by Topic")
    plt.ylabel("Sentiment Score (0-1)")
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"topic_sentiment_{timestamp}.png"))
    print(f"📊 주제별 감정 점수 그래프 저장 완료.")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=topic_df, x="topic", y="avg_sentiment_intensity", palette="viridis")
    plt.title("Average Sentiment Intensity by Topic")
    plt.ylabel("Sentiment Intensity (0-1)")
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"topic_intensity_{timestamp}.png"))
    print(f"📊 주제별 감정 강도 그래프 저장 완료.")

# 기존 함수를 개선된 버전으로 대체
def parallel_search_news(keywords, sources, max_threads=8):
    """병렬로 여러 뉴스 검색을 수행 (안전하게 중단 가능)"""
    results = []

    def process(keyword, source, country):
        try:
            articles = search_expanded_news(keyword, source)
            return [analyze_article_sentiment(article, country, search_keyword=keyword) for article in articles[:5]]
        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생 ({keyword}, {source}): {e}")
            return []

    try:
        with ThreadPoolExecutor(max_threads) as executor:
            futures = []
            for keyword in keywords:
                for source, country in sources.items():
                    futures.append(executor.submit(process, keyword, source, country))

            for future in futures:
                try:
                    results.extend(future.result())  # 결과를 한 번에 추가
                except Exception as e:
                    print(f"⚠️ 병렬 처리 중 오류 발생: {e}")
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다. 안전하게 종료합니다...")
        # 이미 수집된 결과만 반환
        return results

    return results

# 메인 함수를 개선된 버전으로 대체
def main():
    """메인 실행 함수"""
    print("🚀 국가별 및 신문사별 감정 강도 비교 연구를 실행합니다.")
    
    try:
        # 오류 처리 설정
        setup_error_handling()

        # 모든 키워드 사용
        keywords = [kw for topic in TOPIC_KEYWORDS.values() for kw in topic]
        print(f"모든 {len(keywords)}개 키워드를 사용합니다: {keywords}")

        # 모든 신문사 사용
        sources = {}
        # USA 신문사 추가
        for source_name in NEWS_SOURCES["USA"]:
            sources[source_name] = 'US'
            
        # UK 신문사 추가
        for source_name in NEWS_SOURCES["UK"]:
            sources[source_name] = 'UK'
            
        print(f"전체 {len(sources)}개 언론사를 분석합니다: {list(sources.keys())}")

        # 기사 수집 및 분석
        results = parallel_search_news(keywords, sources, max_threads=5)  # 스레드 수 감소

        # 결과 저장 및 시각화
        if results:
            result_folder = get_result_folder()
            timestamp = get_timestamp()
            
            df = pd.DataFrame(results)
            save_results_to_csv(results)
            save_topic_statistics(df, result_folder, timestamp)  # 기존 함수
            save_detailed_statistics(df, result_folder, timestamp)  # 새 함수 추가
            
            # 여기서 종합 통계 함수가 제대로 호출되도록 수정
            save_comprehensive_statistics(df, result_folder, timestamp)
            
            # 모든 시각화 함수를 확실히 호출
            try:
                visualize_sentiment_by_source(df, result_folder, timestamp)
                compare_sentiment_tools(df, result_folder, timestamp)
                plot_topic_sentiments(df, result_folder, timestamp)
                plot_sentiment_over_time(df, result_folder, timestamp) 
                plot_correlation(df, result_folder, timestamp)
                plot_sentiment_distribution(df, result_folder, timestamp)
                plot_sentiment_by_bias_over_time(df, result_folder, timestamp)
                plot_sentiment_by_keyword_over_time(df, result_folder, timestamp)
                generate_wordcloud(df, result_folder, timestamp)
            except Exception as e:
                print(f"시각화 중 오류 발생: {e}")
            
            print(f"✅ 총 {len(results)}개 기사 분석 완료. 결과 저장됨.")
        else:
            print("❌ 분석 결과가 없습니다.")
            
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")

def save_results_to_csv(results, filename=None):
    """분석 결과를 CSV 파일로 저장"""
    if not results:
        debug_print("저장할 결과가 없습니다.")
        return
    
    # 결과 폴더 및 타임스탬프 얻기
    result_folder = get_result_folder()
    timestamp = get_timestamp()
    
    # 파일명 설정
    if filename is None:
        filename = f"sentiment_analysis_results_{timestamp}.csv"
    
    # 전체 경로 생성
    filepath = os.path.join(result_folder, filename)
    
    # DataFrame으로 변환 후 저장
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')  # UTF-8 with BOM for Excel compatibility
    
    print(f"\n💾 분석 결과가 저장되었습니다: {filepath}")
    return filepath

def plot_sentiment_over_time(df, result_folder, timestamp):
    """시간에 따른 감정 점수 변화를 시각화"""
    if not validate_dataframe(df, ['published_at', 'final_sentiment_score'], min_rows=10, purpose="시계열 분석"):
        return
        
    # datetime 변환 및 정렬
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at', 'final_sentiment_score'])
    df = df.sort_values('published_at')
    
    if len(df) < 10:
        print("⚠️ 변환 후 데이터가 부족하여 시계열 분석을 건너뜁니다.")
        return
        
    time_df = df.set_index('published_at')['final_sentiment_score'].resample('W').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(time_df.index, time_df.values, marker='o', linestyle='-')
    plt.title(f'Sentiment Score Over Time (Analysis: {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]})')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score (0-1)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'sentiment_over_time_{timestamp}.png'))
    print(f"📊 시간에 따른 감정 점수 변화 시각화 완료.")

def plot_correlation(df, result_folder, timestamp):
    """감정 분석 도구들 간 상관관계 히트맵"""
    try:
        # 사용 가능한 감정 분석 도구들 확인
        available_tools = []
        tool_labels = []
        
        # 데이터프레임에 있는 도구 컬럼 확인
        if 'vader_score' in df.columns:
            available_tools.append('vader_score')
            tool_labels.append('VADER')
        if 'google_score' in df.columns:
            available_tools.append('google_score')
            tool_labels.append('Google NLP')
        if 'huggingface_score' in df.columns:
            available_tools.append('huggingface_score')
            tool_labels.append('HuggingFace')
        if 'sentiment_intensity_score' in df.columns:
            available_tools.append('sentiment_intensity_score')
            tool_labels.append('SentiStrength')
        
        if len(available_tools) < 2:
            print("⚠️ 상관 관계 분석을 위한 충분한 데이터가 없습니다.")
            return
            
        # 상관 관계 분석
        corr_df = df[available_tools].corr()
        
        # 히트맵 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f",
                    xticklabels=tool_labels, yticklabels=tool_labels)
        plt.title("Correlation Between Sentiment Analysis Tools")
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f"sentiment_correlation_{timestamp}.png"))
        print(f"📊 상관 관계 분석 결과 저장 완료.")
        
    except Exception as e:
        print(f"⚠️ 상관 관계 분석 오류: {e}")

def generate_wordcloud(df, result_folder, timestamp):
    """기사 제목에서 워드 클라우드 생성"""
    try:
        # 모든 기사 제목 합치기
        all_titles = ' '.join(df['title'].tolist())
        
        # 워드 클라우드 생성
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=100,
                             contour_width=1,
                             contour_color='steelblue').generate(all_titles)
        
        # 워드 클라우드 저장
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud from News Titles')
        plt.tight_layout()
        
        wordcloud_image_path = os.path.join(result_folder, f"wordcloud_{timestamp}.png")
        plt.savefig(wordcloud_image_path)
        
        print(f"📊 워드 클라우드 저장 완료: {wordcloud_image_path}")
        
        # 감정 점수에 따른 워드 클라우드 (긍정/부정)
        # 긍정적 기사 (점수 > 0.6)
        positive_df = df[df['final_sentiment_score'] > 0.6]
        if len(positive_df) > 5:  # 최소 5개 이상의 기사가 있을 때만 생성
            positive_titles = ' '.join(positive_df['title'].tolist())
            positive_cloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     max_words=50,
                                     contour_width=1,
                                     contour_color='green').generate(positive_titles)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(positive_cloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud from Positive News Titles')
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f"wordcloud_positive_{timestamp}.png"))
            print(f"📊 긍정 뉴스 워드 클라우드 저장 완료")
        
        # 부정적 기사 (점수 < 0.4)
        negative_df = df[df['final_sentiment_score'] < 0.4]
        if len(negative_df) > 5:  # 최소 5개 이상의 기사가 있을 때만 생성
            negative_titles = ' '.join(negative_df['title'].tolist())
            negative_cloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     max_words=50,
                                     contour_width=1,
                                     contour_color='red').generate(negative_titles)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(negative_cloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud from Negative News Titles')
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f"wordcloud_negative_{timestamp}.png"))
            print(f"📊 부정 뉴스 워드 클라우드 저장 완료")
        
    except Exception as e:
        print(f"⚠️ 워드 클라우드 생성 오류: {e}")

def visualize_results(results):
    """모든 시각화 함수를 호출하는 통합 함수"""
    if not results:
        print("시각화할 결과가 없습니다.")
        return
        
    # 데이터프레임으로 변환
    df = pd.DataFrame(results)
    
    # 결과 폴더 및 타임스탬프 얻기
    result_folder = get_result_folder()
    timestamp = get_timestamp()
    
    # 모든 시각화 함수 호출
    try:
        # 기본 시각화 함수들
        visualize_sentiment_by_source(df, result_folder, timestamp)
        compare_sentiment_tools(df, result_folder, timestamp)
        plot_topic_sentiments(df, result_folder, timestamp)
        
        # 시계열 관련 시각화 함수들
        plot_sentiment_over_time(df, result_folder, timestamp)
        plot_sentiment_by_bias_over_time(df, result_folder, timestamp)
        plot_sentiment_by_keyword_over_time(df, result_folder, timestamp)
        
        # 기타 시각화 함수들
        plot_correlation(df, result_folder, timestamp)  # 도구 상관관계 분석 (중복 제거)
        plot_sentiment_distribution(df, result_folder, timestamp)
        generate_wordcloud(df, result_folder, timestamp)
        
        # 샘플 분석 결과 출력
        sample_article_review(df)
        
        # 종합 통계 저장
        save_comprehensive_statistics(df, result_folder, timestamp)
        
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 모든 시각화와 통계 작업이 완료되었습니다!")
    print(f"📁 결과는 {result_folder} 폴더에 저장되었습니다.")

def extract_domain(url):
    """URL에서 도메인을 추출하는 함수"""
    domain = urlparse(url).netloc
    return domain.replace("www.", "")

def normalize_source_name(source):
    """신문사의 이름을 표준화하여 일치율을 높임"""
    source_mapping = {
        "cnn": "CNN",
        "fox news": "Fox News",
        "the guardian": "The Guardian",
        "nyt": "The New York Times",
        "new york times": "The New York Times",
        "bbc": "BBC News",
        "bbc news": "BBC News",
        "the telegraph": "The Telegraph",
        "reuters": "Reuters",
        "bloomberg": "Bloomberg",
        "the independent": "The Independent",
        "financial times": "Financial Times",
        "ft": "Financial Times",
        "the times": "The Times",
        "daily mail": "Daily Mail",
        "the wall street journal": "The Wall Street Journal",
        "wsj": "The Wall Street Journal",
        "the washington post": "The Washington Post"
    }
    
    source_lower = source.lower()
    for key, value in source_mapping.items():
        if key in source_lower:
            return value
    return source

def plot_sentiment_distribution(df, result_folder, timestamp):
    """감정 점수 분포 그래프"""
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['final_sentiment_score'], bins=30, kde=True, color='blue')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='중립점')
        plt.title('Sentiment Score Distribution')
        plt.xlabel('Sentiment Score (0: Negative, 1: Positive)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f"sentiment_distribution_{timestamp}.png"))
        print(f"📊 감정 점수 분포 그래프 저장 완료.")
        
        # 소스별 분포 그래프도 추가
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='source', y='final_sentiment_score', data=df, palette='Set3')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        plt.title('Sentiment Score Distribution by News Source')
        plt.xlabel('News Source')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f"sentiment_distribution_by_source_{timestamp}.png"))
        print(f"📊 신문사별 감정 점수 분포 그래프 저장 완료.")
        
    except Exception as e:
        print(f"⚠️ 분포 그래프 오류: {e}")

def plot_sentiment_by_bias_over_time(df, result_folder, timestamp):
    """정치 성향별 시간에 따른 감정 점수 변화를 시각화"""
    try:
        if 'published_at' not in df.columns or 'bias' not in df.columns:
            print("⚠️ 정치 성향별 시계열 분석에 필요한 열이 없습니다.")
            return
            
        # datetime 형식으로 변환
        if df['published_at'].dtype == 'object':
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # 결측치 제거 및 정렬
        df_clean = df.dropna(subset=['published_at', 'final_sentiment_score', 'bias'])
        df_clean = df_clean.sort_values('published_at')
        
        if len(df_clean) < 10:
            print("⚠️ 정치 성향별 시계열 분석을 위한 데이터가 부족합니다.")
            return
        
        # 유효한 정치 성향 확인
        biases = df_clean['bias'].unique()
        
        # 그래프 생성
        plt.figure(figsize=(14, 8))
        
        # 1. 정치 성향별 시계열
        plt.subplot(2, 1, 1)
        
        for bias in biases:
            bias_df = df_clean[df_clean['bias'] == bias]
            if len(bias_df) >= 5:
                bias_time_df = bias_df.set_index('published_at')['final_sentiment_score'].resample('W').mean()
                plt.plot(bias_time_df.index, bias_time_df.values, 
                       marker='o', linestyle='-', label=f'{bias.capitalize()}')
        
        plt.title('Sentiment Score Over Time by Political Bias')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score (0-1)')
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Neutral Line')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. 정치 성향별 이동 평균 (추세선)
        plt.subplot(2, 1, 2)
        
        for bias in biases:
            bias_df = df_clean[df_clean['bias'] == bias]
            if len(bias_df) >= 10:  # 이동 평균을 위한 충분한 데이터
                bias_time_df = bias_df.set_index('published_at')['final_sentiment_score']
                # 3일 이동 평균
                plt.plot(bias_time_df.index, bias_time_df.rolling('3D').mean(), 
                       linestyle='-', label=f'{bias.capitalize()} (3-Day MA)')
        
        plt.title('Sentiment Trend by Political Bias (3-Day Moving Average)')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score (0-1)')
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'sentiment_by_bias_over_time_{timestamp}.png'))
        print(f"📊 정치 성향별 시간에 따른 감정 점수 시각화 완료.")
        
        # 3. 주제와 정치 성향의 교차 분석 (히트맵)
        if 'topic' in df_clean.columns or 'search_keyword' in df_clean.columns:
            # 주제 컬럼 없는 경우 키워드에서 주제 추론
            if 'topic' not in df_clean.columns and 'search_keyword' in df_clean.columns:
                keyword_to_topic = {}
                for topic, keywords in TOPIC_KEYWORDS.items():
                    for keyword in keywords:
                        keyword_to_topic[keyword] = topic
                df_clean['topic'] = df_clean['search_keyword'].map(keyword_to_topic)
            
            pivot_df = df_clean.pivot_table(
                index='topic', 
                columns='bias', 
                values='final_sentiment_score',
                aggfunc='mean'
            ).dropna()
            
            if not pivot_df.empty and pivot_df.shape[0] >= 2 and pivot_df.shape[1] >= 2:
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
                plt.title('Average Sentiment Score by Topic and Political Bias')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'topic_bias_heatmap_{timestamp}.png'))
                print(f"📊 주제-정치성향 교차 히트맵 저장 완료.")
        
    except Exception as e:
        print(f"⚠️ 정치 성향별 시계열 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def plot_sentiment_by_keyword_over_time(df, result_folder, timestamp):
    """키워드별 시간에 따른 감정 점수 변화를 시각화"""
    try:
        if 'published_at' not in df.columns or 'search_keyword' not in df.columns:
            print("⚠️ 키워드별 시계열 분석에 필요한 열이 없습니다.")
            return
            
        # datetime 형식으로 변환
        if df['published_at'].dtype == 'object':
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # 결측치 제거 및 정렬
        df_clean = df.dropna(subset=['published_at', 'final_sentiment_score', 'search_keyword'])
        df_clean = df_clean.sort_values('published_at')
        
        if len(df_clean) < 10:
            print("⚠️ 키워드별 시계열 분석을 위한 데이터가 부족합니다.")
            return
        
        # 키워드 카운트
        keyword_counts = df_clean['search_keyword'].value_counts()
        
        # 충분한 데이터가 있는 키워드만 선택 (최소 5개 이상)
        top_keywords = keyword_counts[keyword_counts >= 5].index.tolist()
        
        if len(top_keywords) < 2:
            print("⚠️ 키워드별 시계열 분석을 위한 충분한 키워드가 없습니다.")
            return
        
        # 주제별로 키워드 그룹화
        keyword_to_topic = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                keyword_to_topic[keyword] = topic
        
        # 각 주제별로 시계열 그래프 생성
        topics = set(keyword_to_topic[k] for k in top_keywords if k in keyword_to_topic)
        
        for topic in topics:
            # 해당 주제의 키워드만 추출
            topic_keywords = [k for k in top_keywords if k in keyword_to_topic and keyword_to_topic[k] == topic]
            
            if len(topic_keywords) < 2:
                continue
                
            plt.figure(figsize=(14, 8))
            
            for keyword in topic_keywords:
                keyword_df = df_clean[df_clean['search_keyword'] == keyword]
                if len(keyword_df) >= 5:
                    # 주간 평균으로 리샘플링
                    keyword_time_df = keyword_df.set_index('published_at')['final_sentiment_score'].resample('W').mean()
                    plt.plot(keyword_time_df.index, keyword_time_df.values, 
                           marker='o', linestyle='-', label=f'{keyword}')
            
            plt.title(f'Sentiment Score Over Time for {topic} Keywords')
            plt.xlabel('Date')
            plt.ylabel('Sentiment Score (0-1)')
            plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Neutral Line')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f'sentiment_by_{topic}_keywords_{timestamp}.png'))
            print(f"📊 {topic} 키워드별 시간에 따른 감정 점수 시각화 완료.")
        
        # 키워드별 감정 점수 평균 비교 (막대 그래프)
        plt.figure(figsize=(16, 8))
        
        # 상위 15개 키워드만 표시
        top_n = min(15, len(top_keywords))
        top_n_keywords = top_keywords[:top_n]
        
        # 각 키워드의 평균 감정 점수 계산
        keyword_avg_scores = []
        for keyword in top_n_keywords:
            avg_score = df_clean[df_clean['search_keyword'] == keyword]['final_sentiment_score'].mean()
            keyword_avg_scores.append((keyword, avg_score))
        
        # 감정 점수 내림차순으로 정렬
        keyword_avg_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 막대 그래프 생성
        keywords = [k[0] for k in keyword_avg_scores]
        scores = [k[1] for k in keyword_avg_scores]
        
        bars = plt.bar(keywords, scores, color='skyblue')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral Line')
        
        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Average Sentiment Score by Keyword')
        plt.xlabel('Keyword')
        plt.ylabel('Sentiment Score (0-1)')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'keyword_avg_sentiment_{timestamp}.png'))
        print(f"📊 키워드별 평균 감정 점수 비교 그래프 저장 완료.")
        
    except Exception as e:
        print(f"⚠️ 키워드별 시계열 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def save_comprehensive_statistics(df, result_folder, timestamp):
    """모든 차원(주제/국가/성향/시간)의 통계를 CSV로 저장"""
    try:
        if df.empty:
            print("⚠️ 통계 저장을 위한 데이터가 없습니다.")
            return
            
        # 1. 시간별 통계 저장
        if 'published_at' in df.columns:
            # datetime 형식으로 변환
            if df['published_at'].dtype == 'object':
                df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
                
            # 결측치 제거
            time_df = df.dropna(subset=['published_at', 'final_sentiment_score'])
            
            if not time_df.empty:
                # 일별 통계
                daily_stats = time_df.set_index('published_at')['final_sentiment_score'].resample('D').agg(['mean', 'std', 'count']).reset_index()
                daily_stats.columns = ['date', 'avg_sentiment_score', 'std_sentiment_score', 'article_count']
                
                # 주별 통계
                weekly_stats = time_df.set_index('published_at')['final_sentiment_score'].resample('W').agg(['mean', 'std', 'count']).reset_index()
                weekly_stats.columns = ['week', 'avg_sentiment_score', 'std_sentiment_score', 'article_count']
                
                # CSV로 저장
                daily_stats.to_csv(os.path.join(result_folder, f'daily_sentiment_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
                weekly_stats.to_csv(os.path.join(result_folder, f'weekly_sentiment_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
                print(f"📊 시간별 감정 통계 저장 완료.")
        
        # 2. 정치 성향별 통계 저장
        if 'bias' in df.columns:
            bias_stats = df.groupby('bias').agg({
                'final_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
                'sentiment_intensity_score': ['mean', 'std']
            }).reset_index()
            
            # 컬럼명 재구성
            bias_stats.columns = ['_'.join(col).strip('_') for col in bias_stats.columns.values]
            
            # CSV로 저장
            bias_stats.to_csv(os.path.join(result_folder, f'political_bias_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 정치 성향별 감정 통계 저장 완료.")
        
        # 3. 국가별 통계 저장
        if 'country' in df.columns:
            country_stats = df.groupby('country').agg({
                'final_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
                'sentiment_intensity_score': ['mean', 'std']
            }).reset_index()
            
            # 컬럼명 재구성
            country_stats.columns = ['_'.join(col).strip('_') for col in country_stats.columns.values]
            
            # CSV로 저장
            country_stats.to_csv(os.path.join(result_folder, f'country_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 국가별 감정 통계 저장 완료.")
        
        # 4. 소스별 통계 저장
        if 'source' in df.columns:
            source_stats = df.groupby('source').agg({
                'final_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
                'sentiment_intensity_score': ['mean', 'std']
            }).reset_index()
            
            # 컬럼명 재구성
            source_stats.columns = ['_'.join(col).strip('_') for col in source_stats.columns.values]
            
            # 최소 3개 이상의 기사가 있는 소스만 유지
            filtered_source_stats = source_stats[source_stats['final_sentiment_score_count'] >= 3]
            
            # CSV로 저장
            filtered_source_stats.to_csv(os.path.join(result_folder, f'news_source_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 신문사별 감정 통계 저장 완료.")
        
        # 5. 주제-키워드별 통계 저장
        if 'search_keyword' in df.columns:
            # 키워드별 통계
            keyword_stats = df.groupby('search_keyword').agg({
                'final_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
                'sentiment_intensity_score': ['mean', 'std']
            }).reset_index()
            
            # 컬럼명 재구성
            keyword_stats.columns = ['_'.join(col).strip('_') for col in keyword_stats.columns.values]
            
            # 키워드-주제 매핑
            keyword_to_topic = {}
            for topic, keywords in TOPIC_KEYWORDS.items():
                for keyword in keywords:
                    keyword_to_topic[keyword] = topic
            
            # 주제 컬럼 추가
            keyword_stats['topic'] = keyword_stats['search_keyword'].map(keyword_to_topic)
            
            # 컬럼 순서 재정렬
            cols = keyword_stats.columns.tolist()
            cols.insert(1, cols.pop(-1))  # 'topic' 컬럼을 'search_keyword' 다음으로 이동
            keyword_stats = keyword_stats[cols]
            
            # CSV로 저장
            keyword_stats.to_csv(os.path.join(result_folder, f'keyword_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 키워드별 감정 통계 저장 완료.")
        
        # 6. 국가-정치성향 교차 통계 저장
        if 'country' in df.columns and 'bias' in df.columns:
            cross_stats = df.groupby(['country', 'bias']).agg({
                'final_sentiment_score': ['mean', 'std', 'count'],
                'sentiment_intensity_score': ['mean', 'std']
            }).reset_index()
            
            # 컬럼명 재구성
            cross_stats.columns = ['_'.join(col).strip('_') for col in cross_stats.columns.values]
            
            # CSV로 저장
            cross_stats.to_csv(os.path.join(result_folder, f'country_bias_cross_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 국가-정치성향 교차 통계 저장 완료.")
        
        # 7. 감정 분석 도구 간 비교 통계 저장
        tool_cols = ['vader_score', 'google_score', 'huggingface_score', 'sentiment_intensity_score']
        available_tools = [col for col in tool_cols if col in df.columns]
        
        if len(available_tools) >= 2:
            # 상관관계 계산
            tool_corr = df[available_tools].corr()
            
            # CSV로 저장
            tool_corr.to_csv(os.path.join(result_folder, f'sentiment_tools_correlation_{timestamp}.csv'), encoding='utf-8-sig')
            print(f"📊 감정 분석 도구 간 상관관계 통계 저장 완료.")
            
            # 도구별 평균 및 표준편차
            tool_stats = {tool: {'mean': df[tool].mean(), 'std': df[tool].std()} for tool in available_tools}
            tool_stats_df = pd.DataFrame(tool_stats).T
            tool_stats_df.index.name = 'sentiment_tool'
            tool_stats_df.reset_index(inplace=True)
            
            # CSV로 저장
            tool_stats_df.to_csv(os.path.join(result_folder, f'sentiment_tools_stats_{timestamp}.csv'), index=False, encoding='utf-8-sig')
            print(f"📊 감정 분석 도구별 통계 저장 완료.")
            
        print(f"✅ 모든 차원의 통계 저장이 완료되었습니다.")
        
    except Exception as e:
        print(f"⚠️ 종합 통계 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def plot_tool_correlation(df, result_folder, timestamp):
    """감정 분석 도구 간 상관관계를 히트맵으로 시각화"""
    try:
        # 사용 가능한 감정 분석 도구 확인
        available_tools = []
        tool_labels = []
        
        if 'vader_score' in df.columns:
            available_tools.append('vader_score')
            tool_labels.append('VADER')
        if 'google_score' in df.columns:
            available_tools.append('google_score')
            tool_labels.append('Google NLP')
        if 'huggingface_score' in df.columns:
            available_tools.append('huggingface_score')
            tool_labels.append('HuggingFace')
        if 'sentiment_intensity_score' in df.columns:
            available_tools.append('sentiment_intensity_score')
            tool_labels.append('SentiStrength')
            
        if len(available_tools) < 2:
            print("⚠️ 도구 상관관계 분석을 위한 충분한 데이터가 없습니다.")
            return
            
        # 상관관계 계산
        correlation = df[available_tools].corr()
        
        # 히트맵 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                   xticklabels=tool_labels, yticklabels=tool_labels)
        plt.title('Correlation Between Sentiment Analysis Tools')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'tool_correlation_{timestamp}.png'))
        print(f"📊 감정 분석 도구 간 상관관계 히트맵 저장 완료.")
            
    except Exception as e:
        print(f"⚠️ 도구 상관관계 분석 중 오류 발생: {e}")

def validate_dataframe(df, required_columns, min_rows=5, purpose="분석"):
    """데이터프레임에 필요한 컬럼이 있는지 검증"""
    
    # 데이터프레임이 비어있는 경우
    if df is None or len(df) == 0:
        print(f"⚠️ 데이터프레임이 비어있습니다. {purpose}을(를) 건너뜁니다.")
        return False
        
    # 최소 행 수 확인
    if len(df) < min_rows:
        print(f"⚠️ 데이터가 부족합니다. (필요: {min_rows}개, 현재: {len(df)}개) {purpose}을(를) 건너뜁니다.")
        return False
    
    # 필요한 컬럼 확인
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        # 날짜 컬럼에 대한 대체 확인 ('published_at' 또는 'date' 중 하나 필요)
        if 'published_at' in missing_columns and 'date' in df.columns:
            # date 컬럼이 있으면 published_at으로 복사
            df['published_at'] = df['date']
            missing_columns.remove('published_at')
        
        if 'date' in missing_columns and 'published_at' in df.columns:
            # published_at 컬럼이 있으면 date로 복사
            df['date'] = df['published_at']
            missing_columns.remove('date')
            
        # 여전히 누락된 컬럼이 있다면 오류 메시지 출력
        if missing_columns:
            print(f"⚠️ 필요한 컬럼이 없습니다: {missing_columns}. {purpose}을(를) 건너뜁니다.")
            return False
        
    return True

def preprocess_time_series_data(df):
    """시계열 데이터 전처리 함수"""
    # 날짜 컬럼 처리
    date_columns = ['published_at', 'date']
    date_column = None
    
    # 사용 가능한 날짜 컬럼 찾기
    for col in date_columns:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        # 날짜 컬럼이 없는 경우 현재 날짜로 생성
        print("⚠️ 날짜 컬럼이 없습니다. 현재 날짜를 사용합니다.")
        df['published_at'] = datetime.datetime.now()
        date_column = 'published_at'
    
    # 날짜 형식 변환
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # 결측 날짜 처리
    df = df.dropna(subset=[date_column])
    
    # 날짜순으로 정렬
    df = df.sort_values(date_column)
    
    # 다른 날짜 컬럼이 필요하면 복사
    for col in date_columns:
        if col != date_column and col not in df.columns:
            df[col] = df[date_column]
    
    return df

# 날짜 처리 개선 함수
def ensure_datetime_format(date_str):
    """다양한 형식의 날짜 문자열을 datetime 객체로 변환"""
    if isinstance(date_str, pd.Timestamp) or isinstance(date_str, datetime.datetime):
        return date_str
    
    try:
        return pd.to_datetime(date_str)
    except:
        # 다양한 날짜 포맷 시도
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except:
                continue
                
        # 모든 포맷 시도 실패시
        print(f"날짜 변환 실패: {date_str}")
        return None

def safe_visualization_wrapper(func):
    """시각화 함수를 안전하게 실행하는 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            plt.close('all')  # 모든 열린 그래프 닫기
            return result
        except Exception as e:
            print(f"⚠️ {func.__name__} 실행 중 오류 발생: {e}")
            if plt.gcf().number:
                plt.close('all')  # 열린 figure 닫기
    return wrapper

def save_detailed_statistics(df, result_folder, timestamp):
    """모든 차원의 상세 통계를 저장하는 간단한 함수"""
    
    # 1. 성향별 통계
    if 'bias' in df.columns:
        bias_stats = df.groupby('bias').agg({
            'final_sentiment_score': ['count', 'mean', 'median', 'var', 'std', 'min', 'max'],
            'sentiment_intensity_score': ['mean', 'var', 'std']
        }).reset_index()
        
        # 컬럼명 정리
        bias_stats.columns = ['_'.join(col).strip('_') for col in bias_stats.columns.values]
        
        # 저장
        bias_stats.to_csv(f"{result_folder}/bias_statistics_{timestamp}.csv", 
                        index=False, encoding="utf-8-sig")
        print(f"📊 정치 성향별 통계 저장 완료")
    
    # 2. 도구별 점수 통계
    tool_cols = ['vader_score', 'google_score', 'huggingface_score', 'sentiment_intensity_score']
    available_tools = [col for col in tool_cols if col in df.columns]
    
    if available_tools:
        tool_stats = pd.DataFrame()
        for tool in available_tools:
            stats = df[tool].agg(['count', 'mean', 'median', 'var', 'std']).reset_index()
            stats.columns = ['statistic', tool]
            if tool_stats.empty:
                tool_stats = stats
            else:
                tool_stats = pd.merge(tool_stats, stats, on='statistic')
        
        tool_stats.to_csv(f"{result_folder}/tools_statistics_{timestamp}.csv", 
                       index=False, encoding="utf-8-sig")
        print(f"📊 감정 분석 도구별 통계 저장 완료")
    
    # 3. 키워드별 통계
    if 'search_keyword' in df.columns:
        keyword_stats = df.groupby('search_keyword').agg({
            'final_sentiment_score': ['count', 'mean', 'var', 'std'],
            'sentiment_intensity_score': ['mean', 'var', 'std']
        }).reset_index()
        
        keyword_stats.columns = ['_'.join(col).strip('_') for col in keyword_stats.columns.values]
        keyword_stats.to_csv(f"{result_folder}/keyword_statistics_{timestamp}.csv", 
                          index=False, encoding="utf-8-sig")
        print(f"📊 키워드별 통계 저장 완료")

def check_date_distribution(df):
    """데이터의 날짜 분포를 확인하는 함수"""
    if 'published_at' in df.columns:
        # 날짜 변환
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        date_count = df['published_at'].dt.date.value_counts().sort_index()
        
        print("\n===== 날짜별 기사 수 =====")
        for date, count in date_count.items():
            print(f"{date}: {count}개 기사")
        
        unique_dates = len(date_count)
        print(f"\n총 {unique_dates}개의 고유 날짜가 있습니다.")
        
        if unique_dates < 3:
            print("⚠️ 경고: 시계열 분석을 위한 충분한 날짜 분포가 없습니다.")
        
        # 변환 실패한 날짜 확인
        null_dates = df['published_at'].isnull().sum()
        if null_dates > 0:
            print(f"⚠️ 경고: {null_dates}개의 날짜를 변환하지 못했습니다.")

if __name__ == "__main__":
    main()
    
