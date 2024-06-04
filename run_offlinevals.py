from offlineval_all import main

# SET DEFAULT VALUES

# _RECSYS_MODEL = "ALL"
_FILTER = "PRE" # | "PRE" | "POST" | "None"
_CUSTOM_TRAIN_TEST_SPLIT = "TV" # | None | "is_daytime" | "is_weekday" | "TV" | "WEB" | "MOBILE" |
_INVERSE = False # accepted values: True | False | If this is True then is_daytime means is_nighttime
_OUTPUT_FILENAME = "results"

_TIMES_TO_RUN = 10
_HYPERPARAMS = {'MODEL_PARAM_REGULARIZATION': 0.1, 'MODEL_PARAM_ITERATIONS': 10, 'MODEL_PARAM_FACTORS': 50}
_TOP_N = 10
_TEST_SIZE = 0.25
_NUM_WEEKS = 4
_CONTENT_TYPE_FILTER = "MOVIE" # | "None" | "MOVIE" | "SERIES" |

_RETRIEVE_ALL_RECS = False # | True | False |
_ADJUST_K = 200 # put 0 to adjust every item. inactive without FILTER="POST"

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_param_dict(FILTER=_FILTER, CUSTOM_TRAIN_TEST_SPLIT=_CUSTOM_TRAIN_TEST_SPLIT, INVERSE=_INVERSE, OUTPUT=_OUTPUT_FILENAME, NUM_WEEKS=_NUM_WEEKS, TEST_SIZE=_TEST_SIZE, \
    CONTENT_TYPE_FILTER=_CONTENT_TYPE_FILTER, TOP_N=_TOP_N, RETRIEVE_ALL_RECS=_RETRIEVE_ALL_RECS, HYPERPARAMS=_HYPERPARAMS, TIMES_TO_RUN=_TIMES_TO_RUN, ADJUST_K=_ADJUST_K):
    """
    Parameters:
        CUSTOM_TRAIN_TEST_SPLIT: in [None, 'is_weekday', 'is_weekend', 'is_daytime', 'is_nighttime', 'TV', 'MOBILE', 'WEB']
        INVERSE: True or False
        FILTER: in ["PRE", "POST", "None", None]
        CONTENT_TYPE_FILTER: in [None, "None", "MOVIE", "SERIES"]. If defined, only this content type is kept from raw data.
        TOP_N: the top n recs to be used for evaluation
        RETRIEVE_ALL_RECS: IF TRUE, AVG_RANK_OF_RELEVANTS WILL BE CALCULATED
        HYPERPARAMS: Dict with the key-values you want to override. The rest use the defaults.
    """
    return {"FILTER":FILTER, "CUSTOM_TRAIN_TEST_SPLIT":CUSTOM_TRAIN_TEST_SPLIT, "INVERSE":INVERSE, "OUTPUT_FILENAME":OUTPUT, \
        "TIMES_TO_RUN":TIMES_TO_RUN, "HYPERPARAMS":HYPERPARAMS, "TOP_N":TOP_N, "TEST_SIZE":TEST_SIZE, "NUM_WEEKS":NUM_WEEKS, \
        "CONTENT_TYPE_FILTER":CONTENT_TYPE_FILTER, "RETRIEVE_ALL_RECS":RETRIEVE_ALL_RECS, "ADJUST_K":ADJUST_K}
    

run_params = []


# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="movie_pre_WEB", CUSTOM_TRAIN_TEST_SPLIT="WEB"))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="movie_pre_MOBILE", CUSTOM_TRAIN_TEST_SPLIT="MOBILE"))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="movie_pre_TV", CUSTOM_TRAIN_TEST_SPLIT="TV"))

# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="series_pre_WEB", CUSTOM_TRAIN_TEST_SPLIT="WEB"))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="series_pre_MOBILE", CUSTOM_TRAIN_TEST_SPLIT="MOBILE"))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="series_pre_TV", CUSTOM_TRAIN_TEST_SPLIT="TV"))


# run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="SERIES", OUTPUT="post_series_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))
# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="base_movie_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="pre_movie_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="post_movie_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="base_movie_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="pre_movie_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="post_movie_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="base_movie_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="pre_movie_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="post_movie_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="base_movie_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="pre_movie_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="MOVIE", OUTPUT="post_movie_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True))




# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="SERIES", OUTPUT="base_series_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="pre_series_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="SERIES", OUTPUT="post_series_weekday", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="SERIES", OUTPUT="base_series_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="pre_series_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))
# #---------------
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="SERIES", OUTPUT="post_series_weekend", CUSTOM_TRAIN_TEST_SPLIT="is_weekday", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="SERIES", OUTPUT="base_series_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="pre_series_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="SERIES", OUTPUT="post_series_daytime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=False, NUM_WEEKS=4, TIMES_TO_RUN=10))

# run_params.append(make_param_dict(FILTER="None", CONTENT_TYPE_FILTER="SERIES", OUTPUT="base_series_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))
# run_params.append(make_param_dict(FILTER="PRE", CONTENT_TYPE_FILTER="SERIES", OUTPUT="pre_series_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))
run_params.append(make_param_dict(FILTER="POST", CONTENT_TYPE_FILTER="SERIES", OUTPUT="post_series_nighttime", CUSTOM_TRAIN_TEST_SPLIT="is_daytime", INVERSE=True, NUM_WEEKS=4, TIMES_TO_RUN=10))


for params in run_params:
    main(FILTER=params["FILTER"], CUSTOM_TRAIN_TEST_SPLIT=params["CUSTOM_TRAIN_TEST_SPLIT"], INVERSE=params["INVERSE"], OUTPUT_FILENAME=params["OUTPUT_FILENAME"], NUM_WEEKS=params["NUM_WEEKS"], TEST_SIZE=params["TEST_SIZE"], \
        CONTENT_TYPE_FILTER=params["CONTENT_TYPE_FILTER"], TOP_N=params["TOP_N"], RETRIEVE_ALL_RECS=params["RETRIEVE_ALL_RECS"], HYPERPARAMS=params["HYPERPARAMS"], TIMES_TO_RUN=params["TIMES_TO_RUN"], ADJUST_K=params["ADJUST_K"])