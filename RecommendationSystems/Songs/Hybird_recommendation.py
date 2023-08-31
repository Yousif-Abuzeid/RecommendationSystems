from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

def remove_curly_braces(text):
    return text.replace('{', '').replace('}', '').replace('"','')

def remove_patterns_within_brackets(text):
    pattern = r'\[[^\]]+\]'
    return re.sub(pattern, '', text)

class MusicRecommendationSystem:
    def __init__(self, df_tf,df_collab):
        self.df_tf = df_tf.copy()
        self.df_collab = df_collab.copy()
        self.tfidf_vectorizer = TfidfVectorizer()

    def tfidf_recommendation(self, IDS):
        df = self.df_tf.copy()
        df['text_features'] = (
            df['artist'] + ' ' +  
            df['features'] + ' ' + 
            df['features'] + ' ' +  
            df['artist'] + ' ' +     
            df['artist'] + ' ' +  
            df['tag'] + ' ' + 
            df['tag'] + ' ' + 
            df['title'] + ' ' + 
            df['lyrics'] + ' ' + 
            df['title'] + ' ' + 
            df['artist'] + ' '+
            df['tag'] + ' ' + 
            df['tag'] + ' ' + 
            df['features'] + ' ' +
            df['features'] + ' ' +  
            df['artist'] + ' ' +
            df['artist'] + ' ' +     
            df['artist'] + ' ' +  
            df['artist']
        )
        df['text_features'].fillna('', inplace=True)

        # Fit and transform the text features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_features'])
        texts = df[df['id'].isin(IDS)]['text_features']
        input_tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        similarities = cosine_similarity(input_tfidf_matrix, tfidf_matrix)
        average_similarity = np.mean(similarities, axis=0)
        sorted_indices = average_similarity.argsort()[::-1]
        recommended_artists = df.iloc[sorted_indices[0:20]]

        return recommended_artists

    def collab_recommend(self,user_id,song_list=[]):
        
        df=self.df_collab.copy()

        my_user=df[df['userID']==user_id]

        if(song_list!=[]):
            my_user_latest_songs=song_list

        my_user_latest_songs=my_user.sort_values(by='rating',ascending=False).head(5)

        similar_users=df[df['songID'].isin(my_user_latest_songs['songID'].tolist()) &  (df['userID'] != user_id) &(df['rating'] > 4) ]

        similar_users=df[df['userID'].isin(similar_users['userID'])]

        similar_users_songs=similar_users[~similar_users['songID'].isin(my_user_latest_songs['songID'].tolist())]

        song_options=similar_users_songs[similar_users_songs['rating'] >= 4]

        song_options=song_options.sort_values(by='rating',ascending=False)

        new_df=song_options

        mean_ratings = song_options.groupby('songID')['rating'].mean()

    # Merge the mean_ratings back into the original DataFrame
        new_df = new_df.merge(mean_ratings, left_on='songID', right_index=True, suffixes=('', '_mean'))

        value_counts=new_df['songID'].value_counts()

        new_df['value_counts_col']=new_df['songID'].map(value_counts)

        new_df=new_df.sort_values(by=['value_counts_col','rating_mean'],ascending=False)

        new_df=new_df.drop_duplicates(subset=['songID'])
        new_df=new_df.drop(columns=['userID'])
        return new_df[:5]

        # Rest of the method implementation remains the same
        
# Example usage
df_tf = pd.read_csv('reducedfile.csv')
df_collab=pd.read_csv('songsDataset.csv')
df_collab.rename(columns={"'userID'":'userID'}, inplace=True)
df_collab.rename(columns={"'songID'":'songID'}, inplace=True)
df_collab.rename(columns={"'rating'":'rating'}, inplace=True)
recommendation_system = MusicRecommendationSystem(df_tf,df_collab)

collab_recommendations = recommendation_system.collab_recommend(1)
tfidf_recommendations = recommendation_system.tfidf_recommendation([7539518,124418,3322893,2225459,3402550,1498,7476506])

