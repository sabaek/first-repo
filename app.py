import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, accuracy
from collections import defaultdict
import wordcloud
from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

SEED = 42
reset_seeds(SEED)

df = pd.read_csv('C:/Users/INTEL WIN11PRO(12TH)/Documents/서산데/파이널/data/아모레크롤링_스킨케어(1).csv')

# 제품별 평균 평점과 리뷰 수 계산
product_rating_avg = df.groupby('상품명')['별점'].mean()
product_rating_count = df.groupby('상품명').size()

# 가중 평점 계산
m = product_rating_count.quantile(0.6)
C = product_rating_avg.mean()
product_weighted_rating = (product_rating_count / (product_rating_count + m) * product_rating_avg) + (m / (product_rating_count + m) * C)

# 가중 평점을 데이터프레임에 추가
df['가중평점'] = df['상품명'].map(product_weighted_rating)

# 가상 유저 생성
df['가상유저'] = df['나이'] + ',' + df['성별'] + ',' + df['피부타입'] + ',' + df['피부트러블']

# 각 가상 유저별 리뷰 수 계산
user_review_counts = df['가상유저'].value_counts()

# 가상 유저와 상품명을 ID로 변환
user_to_id = {user: i for i, user in enumerate(df['가상유저'].unique())}
product_to_id = {product: j for j, product in enumerate(df['상품명'].unique())}
df['user_id'] = df['가상유저'].map(user_to_id)
df['product_id'] = df['상품명'].map(product_to_id)

# 가상유저별 총 구매횟수를 계산
user_total_purchase_count = df.groupby('가상유저').size().reset_index(name='총구매횟수')

# 구매횟수를 기반으로 10%씩 묶어 클래스를 생성
user_total_purchase_count['구매_클래스'] = pd.qcut(user_total_purchase_count['총구매횟수'], 10, labels=False)

# 원본 데이터에 구매 클래스 정보 추가
df = pd.merge(df, user_total_purchase_count[['가상유저', '구매_클래스']], on='가상유저', how='left')

train_df, test_df = train_test_split(df,test_size=0.2,random_state=SEED,stratify=df['구매_클래스'])

# Reader 객체 생성
reader = Reader(rating_scale=(0, 5))

# 학습 데이터와 테스트 데이터를 surprise의 데이터 형식으로 변환
train_data_surprise = Dataset.load_from_df(train_df[['user_id', 'product_id', '가중평점']], reader)
trainset = train_data_surprise.build_full_trainset()

# 테스트 데이터를 surprise의 데이터 형식으로 변환
testset = [(row['user_id'], row['product_id'], row['가중평점']) for i, row in test_df.iterrows()]

best_params = {'n_epochs': 100, 'lr_all': 0.005, 'reg_all': 0.2}
# SVD 알고리즘 사용하여 모델 학습
model = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'],random_state=SEED)
model.fit(trainset)

# 테스트 데이터에 대한 예측
predictions = model.test(testset)

# 평가 (RMSE)
rmse = accuracy.rmse(predictions)

id_to_user = {v: k for k, v in user_to_id.items()}
id_to_product = {v: k for k, v in product_to_id.items()}

def get_top_n_recommendations(predictions, n=5):
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        user_info = id_to_user[uid]
        product_name = id_to_product[iid]

        if user_info not in top_n:
            top_n[user_info] = []

        top_n[user_info].append((product_name, est))

    # 정렬, 중복 제거
    for user_info, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        top_n_without_duplicates = []
        for product_name, est in user_ratings:
            if product_name not in seen:
                seen.add(product_name)
                top_n_without_duplicates.append((product_name, est))

        top_n[user_info] = top_n_without_duplicates[:n]

    return top_n

def get_unrated_items(user, df):
    # 사용자가 평가한 아이템들
    rated_items = set(df[df['가상유저'] == user]['상품명'].tolist())
    # 전체 아이템들
    all_items = set(df['상품명'].tolist())
    # 평가하지 않은 아이템들
    unrated_items = all_items - rated_items
    return unrated_items

user_recommendations_with_rated = get_top_n_recommendations(predictions, n=5)

def content_based_recommendation_with_weights(age, gender, skin_type, skin_trouble, top_n=5, weight=0.1):
    # 사용자 정보와 일치하는 리뷰 데이터 필터링
    filtered_df = df[(df['나이'] == age) & (df['성별'] == gender) &
                     (df['피부타입'] == skin_type) & (df['피부트러블'] == skin_trouble)]

    # 상품별 평균 별점 계산
    product_rating_avg = filtered_df.groupby('상품명')['별점'].mean().reset_index()

    # 가중치 적용: 일치하는 특성이 있을 경우, 가중치를 더한다.
    feature_values = {'나이': age, '성별': gender, '피부타입': skin_type, '피부트러블': skin_trouble}
    for feature, feature_value in feature_values.items():
        feature_weight = filtered_df[filtered_df[feature] == feature_value].groupby('상품명')['별점'].count() * weight
        product_rating_avg = pd.merge(product_rating_avg, feature_weight.reset_index().rename(columns={'별점': f'{feature}_weight'}), on='상품명', how='left')

    # 최종 점수 계산 (평균 별점 + 가중치 합)
    product_rating_avg['final_score'] = product_rating_avg['별점'] + product_rating_avg[[f'{feature}_weight' for feature in ['나이', '성별', '피부타입', '피부트러블']]].sum(axis=1)

    # 최종 점수가 높은 상위 N개의 상품 추천
    recommended_products = product_rating_avg.sort_values(by='final_score', ascending=False).head(top_n)['상품명'].tolist()

    return recommended_products

def recommend_products_for_user(age, gender, skin_type, skin_trouble, top_n=5):
    # 가상 유저 ID를 생성
    virtual_user = f"{age},{gender},{skin_type},{skin_trouble}"

    # 가상 유저의 리뷰 수 확인
    user_review_count = df[df['가상유저'] == virtual_user].shape[0]

    # 가상유저별 총 구매횟수를 계산
    user_total_purchase_count = df.groupby('가상유저').size().reset_index(name='총구매횟수')

    # 구매횟수 상위 20%에 해당하는 임계값을 계산
    heavy_user_threshold = user_total_purchase_count['총구매횟수'].quantile(0.8)

    # 리뷰 수 상위 20% 이하인 경우 라이트 유저로 판단
    if user_review_count <= heavy_user_threshold:
        return content_based_recommendation_with_weights(age, gender, skin_type, skin_trouble, top_n=top_n)
    else:
        user_id = user_to_id[virtual_user]
        # CF 기반 추천 수행
        user_recommendations = user_recommendations_with_rated.get(virtual_user, [])
        recommended_products = [product_name for product_name, _ in user_recommendations[:top_n]]
        return recommended_products

def generate_wordcloud(texts):
    stopwords = ['이', '그', '저', '것', '들', '등', '을', '를', '에', '와', '과', '의', '로', '으로', '만', '에서', '게', '으로써',
             '처럼', '하고', '도', '면', '못', '좋', '같아요', '네요', '는데', '다가', '아요', '어요', '습니다', '면서', '많이', '너무',
             '정말', '듯', '때', '고', '게다가', '죠', '거든요', '요', '인데', '더', '해', '해서', '든', '뭐', '하며', '된', '걸', '좀',
             '주', '거', '몇', '또', '한', '되', '같', '보이', '나', '이나', '대', '하', '잘', '되어', '아', '했', '건', '해도', '해보',
             '했어', '하    는', '한다', '하면', '해야', '하게', '하자', '하세', '하고', '하느', '하려', '하였', '하면', '하겠', '하셔', '하십',
             '하세요', '하다가', '사용', '제품', '구매', '피부', '느낌', '효과', '가격', '만족', '구입', '배송', '용량', '가격', '행사', '항상', 
             '구매', '사용', '제품', '처음', '계속', '조금', '생각', '보고', '정도']
    okt = Okt()

    # 리뷰에서 명사 추출
    words = []
    for text in texts:
        nouns = okt.nouns(text)
        nouns = [word for word in nouns if word not in stopwords]
        words.extend(nouns)
    
    # 워드클라우드 생성
    wc = wordcloud.WordCloud(font_path='C:\Windows\Fonts\malgun.ttf', background_color='white', width=800, height=400)
    wc.generate(' '.join(words))
    
    return wc

def chatbot_with_recommendation(df, selected_title):
    model_name = "noahkim/KoT5_news_summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    reviews = df[df['상품명'] == selected_title]['리뷰'].to_list()
    reviews_text = ' '.join(reviews)
    
    inputs = tokenizer(reviews_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=40, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    reviews = df[df['상품명'] == selected_title]
    age = reviews['나이'].mode().values[0]
    types = reviews['피부타입'].mode().values[0]
    trouble = reviews['피부트러블'].mode().values[0]
    product_info = f"이 제품은 {age}, {types}이면서 {trouble} 고민을 가진 사람들이 많이 쓰는 제품입니다."
    
    return summary, product_info

st.sidebar.title('Cosmetic Recommend')
st.sidebar.header('추천받고 싶은 유형을 선택하세요')
if st.sidebar.checkbox("상품명 입력"):
    title = st.sidebar.text_input(label="상품명", value="default value")
    if title:
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in df['상품명'].unique().tolist() if condition(t)]
        
        if not matching_titles:
            st.warning("해당 상품이 없습니다.")
        else:
            st.subheader("\n일치하는 상품명 목록:")
            for idx, matching_title in enumerate(matching_titles, start=1):
                st.text(f"{idx}. {matching_title}")

            selected_idx = st.number_input("상품을 선택하세요 (번호 입력):", min_value=1, max_value=len(matching_titles), value=1)
            selected_title = matching_titles[selected_idx - 1]
            
            summary, product_info = chatbot_with_recommendation(df, selected_title)
            st.subheader(f"{selected_title}의 전체 리뷰 요약:")
            st.write(summary)
            st.write(product_info)


if st.sidebar.checkbox("고객타입 입력"):
    gender = st.sidebar.selectbox("성별",["남성","여성"])
    age = st.sidebar.selectbox("나이",["10대","20대","30대","40대","50대 이상"])
    skintype = st.sidebar.selectbox("피부타입",["복합성","건성","수분부족지성","지성","중성","극건성"])
    skintrouble = st.sidebar.selectbox("피부트러블",["민감성","건조함","탄력없음","트러블","주름","모공","칙칙함","복합성"])
    
    if 'recommend_list' not in st.session_state:
        st.session_state['recommend_list'] = []

    if st.sidebar.button("추천받기"):
        st.header(f"{gender}, {age}, {skintype}, {skintrouble} 타입 고객님께 추천하는 제품입니다.")
        recommend_list = recommend_products_for_user(age, gender, skintype, skintrouble)
        st.session_state['recommend_list'] = recommend_list
        for rec in st.session_state['recommend_list']:
            st.write(rec)

    if st.session_state['recommend_list']:
        selected_option = st.selectbox("궁금한 제품을 선택하세요", st.session_state['recommend_list'])

        st.subheader(f"{selected_option}의 리뷰 동향입니다.")
        # 상품에 대한 리뷰만 필터링
        selected_product_reviews = df[df['상품명'] == selected_option]
        # '리뷰작성날짜' 컬럼을 datetime 형태로 변환
        selected_product_reviews['리뷰작성날짜'] = pd.to_datetime(selected_product_reviews['리뷰작성날짜'])
        # 최근 1년 동안의 리뷰만 필터링
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        recent_reviews = selected_product_reviews[selected_product_reviews['리뷰작성날짜'] > one_year_ago]
        # 리뷰 작성 날짜별로 리뷰 수 계산
        review_trend = recent_reviews.groupby(recent_reviews['리뷰작성날짜'].dt.to_period("M")).size().reset_index(name='리뷰수')
        review_trend['리뷰작성날짜'] = review_trend['리뷰작성날짜'].dt.to_timestamp()
        # 그래프로 표현
        plt.figure(figsize=(10, 6))
        plt.plot(review_trend['리뷰작성날짜'], review_trend['리뷰수'], marker='o')
        plt.title('review trend')
        plt.xlabel('date of review')
        plt.ylabel('number of review')
        plt.grid(True)
        st.pyplot(plt)

        st.subheader(f"{selected_option}의 별점 동향입니다.")
        selected_product_reviews = df[df['상품명'] == selected_option]
        selected_product_reviews['리뷰작성날짜'] = pd.to_datetime(selected_product_reviews['리뷰작성날짜'])
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        recent_reviews = selected_product_reviews[selected_product_reviews['리뷰작성날짜'] > one_year_ago]
        star_rating_trend = recent_reviews.groupby(recent_reviews['리뷰작성날짜'].dt.to_period("M"))['별점'].mean().reset_index(name='평균별점')
        star_rating_trend['리뷰작성날짜'] = star_rating_trend['리뷰작성날짜'].dt.to_timestamp()
        plt.figure(figsize=(10, 6))
        plt.plot(star_rating_trend['리뷰작성날짜'], star_rating_trend['평균별점'], marker='o', color='orange')
        plt.title('star_rating trend')
        plt.xlabel('date of star_rating')
        plt.ylabel('average star_rating')
        plt.ylim(0, 5)
        plt.grid(True)
        st.pyplot(plt)

        # 선택한 제품의 리뷰 데이터 가져오기
        selected_product_reviews = df[df['상품명'] == selected_option]
        
        # 리뷰 텍스트를 기반으로 워드클라우드 생성
        wc = generate_wordcloud(selected_product_reviews['리뷰'].tolist())
        
        # 워드클라우드 표시
        st.image(wc.to_array())
