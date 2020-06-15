import os
import time
import gc
import argparse
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Fuzzy string matching like a boss. It uses Levenshtein Distance to
# calculate the differences between sequences in a simple-to-use package.
from fuzzywuzzy import fuzz


class KnnRecommender:
    """
    Here is an item-based collaborative filtering recommender with
    KNN implemented by sklearn
    """

    def __init__(self, path_movies, path_ratings):
        """
        Initialization

        params:
        path_movies: path to movies dataset
        path_ratings: path to ratings dataset
        """
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_threshold = 0
        self.user_rating_threshold = 0
        self.model = NearestNeighbors()

    def set_filter_params(self, movie_rating_threshold, user_rating_threshold):
        """
        set rating freq threshold to filter less-known movies and less active users
        params:
        -------
        movie_rating_threshold: int, is the min number of ratings received by users
        user_rating_threshold: int, is the min number of ratings a user gives

        """
        self.movie_rating_threshold = movie_rating_threshold
        self.user_rating_threshold = user_rating_threshold

    def set_models_params(self, n_neighbors, algo, metric, n_jobs=None):
        """
        set model params for NeirestNeighbors algorithm
        params:
        -------
        n_neighbors: int, optional(default = 5)

        algo: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional

        metric: string or callable, default 'minkowski', or one of
                ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']


        n_jobs: int or none, optional (default=None)
        """

        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algo,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):
        """
        prepare data for recommender

        1. movie-user scipy sparse matrix
        2. hashmap of movie of row index in movie-user scipy sparse matrix

        """
        # read data
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})

        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        # filter data
        ###FOR MOVIES
        df_movies_count = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])

        popular_movies = list(set(df_movies_count.query('count >= @self.movie_rating_threshold').index))
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        ###FOR USERS
        df_users_count = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_count.query('count >= @self.user_rating_threshold').index))
        users_filter = df_ratings.userId.isin(active_users).values

        ## new one
        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        # create mapper from movie title to index

        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))

        }
        # trasform it now to scipy sparse matrix

        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        # now we can delete non needed data
        del df_movies, df_movies_count, df_users_count
        del df_ratings, df_ratings_filtered, movie_user_mat
        gc.collect()
        return movie_user_mat_sparse, hashmap

    # this function use the Leveinshtein algorithm(or distance edition algo)
    def _fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None

        Params:
        ----------
        hashmap: dict, map movie title name to index of the movie in data


        fav_movie: str, name of user input movie


        return
        ----------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, index in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, index, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print("Sorry there is no match!!")
        else:
            print("Found possible matches in db: " +
                  "{0}\n".format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   fav_movie, n_recommendations):

        """
        return top n similar movie recommendations based on user's input
        movie

        params:
        -------
        model: sklearn model, knn
        data: movie_user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations

        return
        -------
        list of top n similar movie recommendations
        """
        # fit data
        model.fit(data)
        # get input movie index
        print('Your input is: ', fav_movie)
        index = self._fuzzy_matching(hashmap, fav_movie)
        # inference
        print('Wait until recommendation system finish his job!')
        print(10 * '.', end='\n')
        # let's time it
        t_0 = time.time()
        distances, indices = model.kneighbors(
            data[index],
            n_neighbors=n_recommendations + 1)
        # get list of raw index of recommendations

        raw_recommends = sorted(list(zip(indices.squeeze().tolist(),
                                         distances.squeeze().tolist())),
                                key=lambda x: x[1])[:0:-1]
        print('It took : {:.2f} seconds'.format(time.time() - t_0))

        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        """
        make top n recommendations

        params:
        -------
        fav_movie: str, user's input movie

        n_recommendations: int, top n recommendations

        """
        # get data

        movie_user_mat_sparse, hashmap = self._prep_data()

        # get recommendations
        raw_recommends = self._inference(self.model,
                                    movie_user_mat_sparse,
                                    hashmap,
                                    fav_movie,
                                    n_recommendations)

        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('recommendations for {}'.format(fav_movie))
        for i, (index, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_hashmap[index], dist))


# in order to make things easy on console let's use a parser
def parse_args():
    """
    The argparse module makes it easy to write user-friendly
    command-line interfaces.
    It parses the defined arguments from the sys.argv.
    The argparse module also automatically generates help and usage messages,
    and issues errors when users give the program invalid arguments.
    :return:
    parsed
    """
    parser = argparse.ArgumentParser(
        prog='Movie Recommender',
        description='Run KNN Movie Recommender')

    parser.add_argument('--path', nargs='?', default='ml-latest-small',
                        help='provide movies filename')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='input path to the data')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='Spider-Man',
                        help='please provide your favorite movie name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')

    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    movie_name = args.movie_name
    top_n = args.top_n

    # initialize recommender system

    recommender = KnnRecommender(os.path.join(data_path, movies_filename),
                                 os.path.join(data_path, ratings_filename))

    recommender.set_filter_params(50, 50)
    recommender.set_models_params(20, 'brute', 'cosine', -1)

    recommender.make_recommendations(movie_name, top_n)

