from django.shortcuts import render
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from imdb import IMDb
# Create your views here.
def recommendation(request):
    return render(request, 'recommendation.html')


ia = IMDb()

def recommend_movies(request):
    df = pd.read_csv("IMDb_All_Genres_etf_clean1.csv", encoding='latin-1')
    features = ['Director', 'Actors', 'main_genre', 'side_genre']
    for f in features:
        df[f] = df[f].fillna('')
    df['Genre'] = df['main_genre'] + ', ' + df['side_genre']
    df['Combined'] = df['Director'] + ', ' + df['Actors'] + ', ' + df['Genre']

    data = df.groupby(['Genre']).agg({'Combined': list, 'Movie_Title': list, }).apply(list).reset_index()
    titles = data["Movie_Title"].tolist()

    te = TransactionEncoder()
    te_ary = te.fit(titles).transform(titles)
    df_end = pd.DataFrame(te_ary, columns=te.columns_)

    apriori_frequent_itemsets = apriori(df_end, min_support=0.001, max_len=2, use_colnames=True)

    rules = association_rules(apriori_frequent_itemsets, metric="lift", min_threshold=0.01)

    input_movie = request.POST.get('movie_name')
    if input_movie:
        recommended_movies = rules[rules["antecedents"].apply(lambda x: input_movie in str(x))].sort_values(
            ascending=False, by='lift')

        recommended_movies = recommended_movies.groupby(['antecedents', 'consequents'])[['lift']].max().sort_values(
            ascending=False, by='lift')

        movie_titles = recommended_movies.index.get_level_values(1).tolist()

        movie_titles_clean = []
        movie_posters = []
        for mt in movie_titles:
            if len(movie_titles_clean) == 18:
                break
            title = str(mt)
            title = title.replace("frozenset({'", "")
            title = title.replace('frozenset({"', "")
            title = title.replace('"})', "")
            title = title.replace("'})", "")
            movie_titles_clean.append(title)

            search_results = ia.search_movie(title)
            if search_results:
                movie = search_results[0]
                ia.update(movie)
                poster_url = movie.get('full-size cover url')
                movie_posters.append(poster_url)
            else:
                movie_posters.append(None)

        search_results = ia.search_movie(input_movie)
        if search_results:
            input_movie_obj = search_results[0]
            ia.update(input_movie_obj)
            input_movie_poster = input_movie_obj.get('full-size cover url')
        else:
            input_movie_poster = None

        if movie_titles_clean:
            return render(request, 'recommendation.html',
                          {'movies': zip(movie_titles_clean[:18], movie_posters[:18]), 'input_movie': input_movie,
                           'input_movie_poster': input_movie_poster, 'has_recommendations': True})
        else:
            return render(request, 'recommendation.html',
                          {'movies': [], 'input_movie': input_movie, 'input_movie_poster': input_movie_poster,
                           'error_msg': 'No recommendations found.'})
