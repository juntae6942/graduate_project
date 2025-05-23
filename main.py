from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


def classify_iso_tempo(tempo):
    if tempo < 70:
        return 'Stage 1'
    elif tempo < 80:
        return 'Stage 2'
    elif tempo < 90:
        return 'Stage 3'
    elif tempo < 100:
        return 'Stage 4'
    elif tempo < 110:
        return 'Stage 5'
    elif tempo < 120:
        return 'Stage 6'
    else:
        return 'Stage 7'


df = pd.read_csv('../datasets/tempo_upper_60.csv')
df['genre_list'] = df['track_genre'].apply(lambda x: [g.strip() for g in x.split(',')])

# 2. MultiLabelBinarizer로 인코딩
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])

# 3. 결과 DataFrame에 붙이기
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)
df['iso_stage'] = df['tempo'].apply(classify_iso_tempo)
features = ['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'] + list(mlb.classes_)

app = FastAPI()


class Input(BaseModel):
    features: List[float]
    genres: List[str]
    stage: int


@app.post("/recommendations")
def predict(data: Input):
    encoded = mlb.transform([data.genres])[0]
    all_features = np.array(data.features + encoded.tolist())

    # 입력 벡터 전처리
    input_df = pd.DataFrame([all_features], columns=features)

    recommendations_list = []

    # 현재 stage부터 Stage 1까지 반복 (역순)
    for stage in range(data.stage, 0, -1):
        stage_str = f'Stage {stage}'
        iso_stage = df[df['iso_stage'] == stage_str].copy()

        if iso_stage.empty:
            continue

        scaler = MinMaxScaler()
        current_scale = scaler.fit_transform(iso_stage[features])
        scaled = scaler.transform(input_df)
        similarity = cosine_similarity(scaled, current_scale)[0]

        iso_stage['similarity'] = similarity
        top_n = iso_stage.sort_values(by='similarity', ascending=False).head(2)  # 각 stage에서 상위 3개
        recommendations_list.append(top_n)

    # 모든 추천을 하나의 DataFrame으로 합치기
    all_recommendations = pd.concat(recommendations_list)
    all_recommendations = all_recommendations.sort_values(by='similarity', ascending=False)

    selected_columns = ["track_name", "artist_name", "similarity", "artwork_url", "track_url", "iso_stage"]
    return all_recommendations[selected_columns].to_dict(orient="records")
