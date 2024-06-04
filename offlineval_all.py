import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pyarrow.parquet as pq
from loguru import logger

from pathlib import Path
import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")
from collections import defaultdict
import sys
import os
from typing import Union, Optional
import random

from sklearn.model_selection import train_test_split

from utils import create_proportions_dict, adjust_for_context, appname_to_device

# Configure logging to file
logger.add("logfile.log", format="{time} {level} {message}", level="DEBUG")

class StreamToLogger:
    def __init__(self, level="DEBUG"):
        self.level = level

    def write(self, message):
        if message.strip() != "":
            logger.log(self.level, message.strip())

    def flush(self):
        pass

# Redirect stdout and stderr to Loguru
sys.stdout = StreamToLogger("INFO")
sys.stderr = StreamToLogger("ERROR")

def main(CUSTOM_TRAIN_TEST_SPLIT: Optional[str], INVERSE: bool, FILTER: Union[bool,str], OUTPUT_FILENAME: str, NUM_WEEKS=2, TEST_SIZE=0.25, CONTENT_TYPE_FILTER=None, \
    TOP_N=10, RETRIEVE_ALL_RECS=False, ADJUST_K=0, TIMES_TO_RUN=10, HYPERPARAMS={'MODEL_PARAM_REGULARIZATION': 0.1, 'MODEL_PARAM_ITERATIONS': 10, 'MODEL_PARAM_FACTORS': 50}):
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

    if __name__ != "__main__":
        logger.success("offlineval.py was executed via import")
    logger.critical(f"starting experiment: {OUTPUT_FILENAME}")

    _HYPERPARAMS = {'MODEL_PARAM_REGULARIZATION': 0.01, 'MODEL_PARAM_ITERATIONS': 10, 'MODEL_PARAM_FACTORS': 50}
    for key, value in HYPERPARAMS.items():
        _HYPERPARAMS[key] = value

    def set_device(dataset):
        dataset["device"] = dataset["appName"].map(appname_to_device)
        return dataset

    def is_weekday(dataset):
        dataset["is_weekday"] = dataset["firstStart"].map(lambda x: True if x.weekday() in [0, 1, 2, 3] else False)
        return dataset

    def is_daytime(dataset, morning_cutoff='04:00', nighttime_cutoff='18:00'):
        dataset["is_daytime"] = (pd.to_datetime(morning_cutoff).time() < dataset['firstStart'].dt.time) & (dataset['firstStart'].dt.time < pd.to_datetime(nighttime_cutoff).time())
        return dataset

    def transform(dataset):
        """ Rename columns """
        dataset.rename(columns={'durationSec': 'playingTime'}, inplace=True)
        dataset.drop(columns="userId", inplace=True)
        dataset.rename(columns={'profileId': 'userId'}, inplace=True)

        dataset.loc[:, "itemId"] = dataset.apply(lambda row: row["categoryId"] if not row["categoryId"] == "" else row["assetId"], axis=1)

        return dataset

    def custom_train_test_split(df, test_size=TEST_SIZE):
        # Mask of rows that are from our target context
        if CUSTOM_TRAIN_TEST_SPLIT in ["is_weekday", "is_daytime"]:
            context_data = df[df[CUSTOM_TRAIN_TEST_SPLIT] != INVERSE]
        else:
            context_data = df[df["device"]==CUSTOM_TRAIN_TEST_SPLIT]

        # Identify invalid indices - rows where userId and itemId combination occur in both contexts
        invalid_indices = set()
        context = CUSTOM_TRAIN_TEST_SPLIT if CUSTOM_TRAIN_TEST_SPLIT in ["is_weekday", "is_daytime"] else "device"
        grouped = df.groupby(['userId', 'itemId']) 
        # what if we first group by user id and make user_items dict so we can skip df.iterrows() below
        # then group by itemId to make invalid indices
        for (user_id, item_id), group in grouped:
            if len(group[context].unique()) > 1:
                invalid_indices.update(group.index)

        # Create a dictionary for all user-item interactions
        user_items = defaultdict(set)
        for idx, row in df.iterrows():
            user_items[row['userId']].add(row['itemId'])

        # Choose test indices ensuring at least one other item per user remains in train
        test_candidates = [idx for idx in context_data.index if idx not in invalid_indices]
        np.random.shuffle(test_candidates)
        valid_test_indices = []
        remaining_users_items = dict(user_items)  # Copy to modify as potential test items are selected

        for idx in test_candidates:
            user_id = context_data.loc[idx, 'userId']
            item_id = context_data.loc[idx, 'itemId']

            # Ensure theres at least one other item for this user thats not the current item
            if len(user_items[user_id] - {item_id}) > 0:
                valid_test_indices.append(idx)
                # Assume this item will be in the test set and remove it from the 'remaining' set
                remaining_users_items[user_id].remove(item_id)

            if len(valid_test_indices) == int(test_size * len(df)):
                break

        if len(valid_test_indices) < int(test_size * len(df)):
            logger.critical("Not enough rows meet the condition to form a full test set of desired size.")

        test_mask = df.index.isin(valid_test_indices)
        test = df.loc[test_mask]
        train = df.loc[~test_mask]
        return train, test



    def agg(dataset):
        """ Aggregate on itemId """
        
        dataset = dataset \
                            .groupby(['userId', 'itemId']) \
                            .agg({'playingTime': 'sum'}) \
                            .reset_index()
        return dataset

    def agg_with_context(dataset, context):
        """ Aggregate on itemId and context (accepted values: 'is_weekday' or 'is_daytime') """
        
        dataset = dataset \
                            .groupby(['userId', 'itemId', context]) \
                            .agg({'playingTime': 'sum'}) \
                            .reset_index()
        return dataset


    def clean(dataset):

        """ Filter out rows with empty keys """
        invalid_rows = (dataset['userId'].str.len() == 0) | \
            (dataset['itemId'].str.len() == 0)

        if invalid_rows.sum() == 0:
            return dataset
        
        logger.warning("rows with invalid keys: {}".format(invalid_rows.sum()))
        
        dataset = dataset[~invalid_rows]
        return dataset

    def ceiling(dataset, max_score):
        """ Set score ceiling """

        mask = dataset['playingTime'] > max_score
        dataset.loc[mask, 'playingTime'] = max_score
        return dataset

    def floor(dataset, cutoff_score):
        """ Filter out scores below threshold """

        mask = dataset['playingTime'] < cutoff_score
        dataset = dataset[~mask].copy() 
        return dataset

    def to_category(dataset):
        """ Convert to category codes, return altered df and reverse mappings for users and items"""
        logger.info("converting to category")

        # Convert to category datatype
        dataset['userId'] = dataset['userId'].astype("category")
        dataset['itemId'] = dataset['itemId'].astype("category")

        # Create reverse mappings
        users = dict(enumerate(dataset['userId'].cat.categories))
        items = dict(enumerate(dataset['itemId'].cat.categories))
        logger.info("number of unique items after filtering, before train/test split: " + str(len(items)))

        # Convert to category codes
        dataset["userId"] = dataset["userId"].cat.codes
        dataset["itemId"] = dataset["itemId"].cat.codes

        return dataset, users, items

    mondays = [2, 9, 16, 23, 30]
    tuesdays = [3, 10, 17, 24, 31]
    wednesdays = [4, 11, 18, 25]
    thursdays = [5, 12, 19, 26]
    fridays = [6, 13, 20, 27]
    saturdays = [7, 14, 21, 28]
    sundays = [1, 8, 15, 22, 29]

    weekdays={"mondays": mondays[:NUM_WEEKS], "tuesdays":tuesdays[:NUM_WEEKS], "wednesdays":wednesdays[:NUM_WEEKS], "thursdays":thursdays[:NUM_WEEKS]}
    weekends={"fridays":fridays[:NUM_WEEKS], "saturdays":saturdays[:NUM_WEEKS], "sundays":sundays[:NUM_WEEKS]}

    weekday_filenames = []
    for days in weekdays.values():
        for day in days:
            weekday_filenames.append("day"+str(day)+".parquet")

    weekend_filenames = []
    for days in weekends.values():
        for day in days:
            weekend_filenames.append("day"+str(day)+".parquet")

    data = pd.DataFrame()
    for i, selected_day in enumerate(weekday_filenames+weekend_filenames): 
        print(f"loading data from day {i+1} out of {len(weekend_filenames+weekday_filenames)}")
        day = pq.read_table(Path(str(os.getcwd())).parent / "data/october_daily" / selected_day).to_pandas()
        data = pd.concat([data, day], axis=0)
        
    #dropping kids content
    data = data[data["kids"] == False]
    if CONTENT_TYPE_FILTER not in ["None", None]:
        data = data[data["contentType"] == CONTENT_TYPE_FILTER]

    # -------------------DATA TRANSFORMATIONS
    data = transform(data)
    num_items_loaded = data["itemId"].nunique() # MOVIES 2 WEEKS = 3122 --- 4 weeks MOVIES = 3514. --- 2 weeks SERIES = 994 --- 4 weeks SERIES = 1002
    logger.info("applying context-awareness measures")
    if CUSTOM_TRAIN_TEST_SPLIT == "is_weekday":
        data = is_weekday(data)
    elif CUSTOM_TRAIN_TEST_SPLIT == "is_daytime":
        data = is_daytime(data)
    elif CUSTOM_TRAIN_TEST_SPLIT in ["TV", "MOBILE", "WEB"]:
        data = set_device(data)

    # ----------- optional: prefiltering for the given context
    if FILTER == "PRE" and CUSTOM_TRAIN_TEST_SPLIT in ["is_weekday", "is_daytime"]:
        data = data[data[CUSTOM_TRAIN_TEST_SPLIT] != INVERSE].copy() #.copy() so that the df becomes an independent object, not just a view of the unfiltered raw data
    elif FILTER == "PRE" and CUSTOM_TRAIN_TEST_SPLIT in ["TV", "WEB", "MOBILE"]:
        data = data[data["device"]==CUSTOM_TRAIN_TEST_SPLIT].copy() #.copy() so that the df becomes an independent object, not just a view of the unfiltered raw data

    data, users, items = to_category(data)

    logger.info("aggregating data...")
    if CUSTOM_TRAIN_TEST_SPLIT in ["is_weekday", "is_daytime"]:
        data = agg_with_context(data, CUSTOM_TRAIN_TEST_SPLIT)
    elif CUSTOM_TRAIN_TEST_SPLIT in ["TV", "WEB", "MOBILE"]:
        data = agg_with_context(data, "device")
    else:
        data = agg(data)

    data = clean(data)

    # ------------ optional: creating dict with proportion of total watchtime that stems from current vs opposite context for every item.
    #  used in post-filtering for weighting confidence scores.
    propdict = "not created"
    if FILTER == "POST":
        propdict = create_proportions_dict(data, CUSTOM_TRAIN_TEST_SPLIT, INVERSE)
        avg_proportion = np.mean(list(value for value in propdict.values()))
        logger.info("average watchtime proportion of chosen context: " + str(avg_proportion))

    upper_threshold_score = 10000#5400
    lower_threshold_score = 500
    data = floor(ceiling(data, upper_threshold_score), lower_threshold_score)

    logger.info("rows in dataset after pre-processing, before train/test split: " + str(len(data)))

    # ---- WRITING EXPERIMENT PARAMETERS TO FILE
    towrite = []
    # towrite.append("RECSYS_MODEL: " + str(RECSYS_MODEL))
    towrite.append("TOP_N: " + str(TOP_N))
    if FILTER == "POST": towrite.append("ADJUST_K: " + str(ADJUST_K))
    if propdict != "not created": towrite.append("AVG_PROPORTION: " + str(avg_proportion))
    towrite.append("UNSPLIT_DF_LEN: " + str(len(data)))

    for key, value in _HYPERPARAMS.items():
        towrite.append(f"{key}: {value}")
  
    towrite.append("CUSTOM_TRAIN_TEST_SPLIT: " + str(CUSTOM_TRAIN_TEST_SPLIT))
    towrite.append("INVERSE: " + str(INVERSE))
    towrite.append("FILTER: " + str(FILTER))
    towrite.append("NUM_WEEKS: " + str(NUM_WEEKS))
    towrite.append("CONTENT_TYPE_FILTER: " + str(CONTENT_TYPE_FILTER))


    with open(f"results/{OUTPUT_FILENAME}.txt", "a") as file:
        file.write("\n")
        file.write("----- experiment parameters ------")
        for exp_param in towrite:
            file.write("\n")
            file.write(exp_param)

    for run_iterator in range(TIMES_TO_RUN):
        logger.success("RUN ITERATION: " + str(run_iterator+1))
        logger.info("performing train/test split")

        if CUSTOM_TRAIN_TEST_SPLIT:
            train_data, test_data = custom_train_test_split(data)
            logger.info("train size: " + str(len(train_data)) + " , test size: " + str(len(test_data)) + ". test proportion: " + str(round(len(test_data)/(len(train_data) + len(test_data)), 3)))
            logger.info("completed custom train/test split")

            # Identify rows in 'test' that have a combination of 'userId' and 'itemId' also present in 'train'
            common_rows = test_data.merge(train_data[['userId', 'itemId']], on=['userId', 'itemId'], how='inner')
            logger.info(f"moving {len(common_rows)} rows from test to train. these had a combination of 'userId' and 'itemId' also present in 'train'.")
            # Append these rows to the 'train' dataframe
            train_data = pd.concat([train_data, common_rows], ignore_index=True)

            # Remove these rows from the 'test' dataframe
            test_data = test_data.merge(train_data[['userId', 'itemId']], on=['userId', 'itemId'], how='left', indicator=True)
            test_data = test_data[test_data['_merge'] == 'left_only'].drop(columns=['_merge'])

            # Aggregate again since last time custom aggregated by grouping on context too (this time groups only on userId and itemId)
            # No need to aggregate the test data again as it contains only rows from one context
            pre_agg_len = len(train_data)
            train_data = agg(train_data)
            post_agg_len = len(train_data)
            logger.info(f"len(train) before re-aggregating = {pre_agg_len}")
            logger.info(f"len(train) after re-aggregating = {post_agg_len}")

            if pre_agg_len > post_agg_len:
                #finding rows to drop from test_data if the proportion got messed up after re-aggregation of train_data
                n_rows_to_drop = round((len(test_data) - TEST_SIZE * (len(train_data) + len(test_data))) / (1-TEST_SIZE))
                if n_rows_to_drop > 0:
                    test_data = test_data.drop(test_data.sample(n=n_rows_to_drop).index)
                else:
                    logger.critical("COULD NOT DROP TEST ROWS AS TEST ROW PROPORTION IS STILL TOO SMALL AFTER RE-AGG OF TRAIN_DATA")
                logger.info("AFTER DROPPING TEST ROWS. train size: " + str(len(train_data)) + " , test size: " + str(len(test_data)) + ". test proportion: " + str(round(len(test_data)/(len(train_data) + len(test_data)), 3)))

            # Need to re-apply ceiling to train data after combining playtime from both contexts
            train_data = ceiling(train_data, upper_threshold_score)


        else:
            train_data, test_data = train_test_split(data, test_size=TEST_SIZE)
        logger.info("train size: " + str(len(train_data)) + " , test size: " + str(len(test_data)) + ". test proportion: " + str(round(len(test_data)/(len(train_data) + len(test_data)), 3)))


        #------------------INITIALIZE AND TRAIN PREDICTIVE MODEL

        # if RECSYS_MODEL == "ALS":
        interaction_matrix = sparse.coo_matrix(
            (train_data['playingTime'].astype(np.float32),
            (train_data['userId'],
            train_data['itemId']))
        ).tocsr()

        model = implicit.als.AlternatingLeastSquares(factors=_HYPERPARAMS["MODEL_PARAM_FACTORS"], calculate_training_loss=True, num_threads=0)

        # Set hyperparameters
        model.regularization = _HYPERPARAMS["MODEL_PARAM_REGULARIZATION"]
        model.iterations = _HYPERPARAMS["MODEL_PARAM_ITERATIONS"]

        logger.info("training model...")
        model.fit(interaction_matrix, show_progress=False)

        #---------- RETRIEVE RECS
        if RETRIEVE_ALL_RECS:
            top_n = interaction_matrix.shape[1]
        elif FILTER == "POST":
            top_n = ADJUST_K
        else:
            top_n = TOP_N
        logger.info(f"retrieving {top_n} recs per user")
        user_ids = list(set(test_data["userId"]) & set(train_data["userId"])) #[:50] # uncomment to dwarf the runtime to check recs and scores for just a few users

        ids, scores = model.recommend(user_ids, interaction_matrix[user_ids], N=top_n)
        
        num_rec_rows = top_n * len(ids)
        
        # elif RECSYS_MODEL == "POPULAR":
        user_seen_items = train_data.groupby('userId')['itemId'].agg(list).to_dict()
        item_popularity = train_data['itemId'].value_counts().sort_values(ascending=False)

        # Function to recommend top N popular items that the user hasn't seen yet
        def recommend_popular_items(user_ids, user_seen_items, item_popularity, n):
            recommendations = {}
            prepared_for_rec_df = []
            for userId in user_ids:
                # Filter popular items to find ones the user hasn't seen
                recommended_items = [item for item in item_popularity.index if item not in user_seen_items[userId]][:n]
                recommendations[userId] = recommended_items

                # ranks = list(range(len(recommended_items)))
                # prepared_for_rec_df.append((userId, recommended_items, [i+1 for i in ranks]))
            return recommendations

        # user_ids = list(set(test_data["userId"]) & set(train_data["userId"]))
        # # top_n = TOP_N
        if FILTER == "POST":
            popular_recs = recommend_popular_items(user_ids, user_seen_items, item_popularity, ADJUST_K)
        else:
            popular_recs = recommend_popular_items(user_ids, user_seen_items, item_popularity, top_n)


        # -------- MAKE GROUND TRUTH DATAFRAME
        logger.info("creating ground_truth dataframe for evaluation")

        ground_truth_df = test_data.sort_values(by=["userId", "playingTime"], ascending=[True, False])
        ground_truth_df = ground_truth_df.groupby("userId").agg({'itemId': list, 'playingTime': list}).reset_index()
        logger.info("Ground truth length before dropping userids not present in train_data: " + str(len(ground_truth_df)))
        ground_truth_df = ground_truth_df[ground_truth_df["userId"].isin(user_ids)] # dropping userids that arent present in training data.
        logger.info("Ground truth length after dropping userids not present in train_data: " + str(len(ground_truth_df)))
        ground_truth_df['is_prediction'] = False
        ground_truth_df.rename(columns={'itemId': 'item_code', 'userId': 'user_code', 'playingTime': 'score'}, inplace=True)

        # ------- MAKE RECS DATAFRAME
        logger.info("creating recommendations dataframe for evaluation")

        if FILTER == "POST":
            ALS_recommendations = {}
            for user_idx, user_code in enumerate(user_ids):
                item_codes = ids[user_idx]
                conf_scores = scores[user_idx]
                _itemscores = []
                for item_code, score in zip(item_codes, conf_scores):
                    _itemscores.append((item_code, np.float32(score)))
                ALS_recommendations[user_code] = _itemscores

        else:
            # if RECSYS_MODEL == "ALS":
                # Preallocate a structured numpy array
            data_type = [('user_code', int), ('item_code', int), ('score', np.float32), ('is_prediction', bool)]
            rec_array = np.zeros(num_rec_rows, dtype=data_type)

            # Fill the array
            idx = 0
            for user_idx, user_code in enumerate(user_ids):
                item_codes = ids[user_idx]
                conf_scores = scores[user_idx]
                for item, score in zip(item_codes, conf_scores):
                    rec_array[idx] = (user_code, item, np.float32(score), True)
                    idx += 1

            rec_df = pd.DataFrame(rec_array)
            rec_df.sort_values(by=['user_code', 'score'], ascending=[True, False], inplace=True)

            # Group by 'user_code' and aggregate 'item_code' and 'score' into lists
            rec_df = rec_df.groupby('user_code').agg({
                'item_code': list,
                'score': list
            }).reset_index()

            rec_df["is_prediction"] = True

        # breakpoint() #-------------------- this breakpoint is good for checking recs and ground truth manually for specific users. to check the postfiltered recs: postfiltered = adjust_for_context(rec_df, propdict, k=ADJUST_K)

        iterations = 1 #if FILTER != "POST" else 3 if TOP_N==5 and ADJUST_K==50 else 2 # if this is a post-filter experiment: evaluate once without and then again with post-filtering. otherwise just evaluate once.
        for _iteration in range(iterations):
            if FILTER == "POST":
                if _iteration == 0:
                    ADJUSTED = "RAN BY _ALL" #"NO"
                elif _iteration == 1:
                    ADJUSTED = "YES"
                elif _iteration == 2:
                    ADJUSTED = "AGAIN_BUT_K0"
            #     logger.warning("evaluating with postfiltering: " + ADJUSTED)
            # if FILTER == "POST":  #_iteration == 1: #if this is the second run, so we have already evaluated the results without post-filtering
            #     rec_df = adjust_for_context(rec_df, propdict, k=ADJUST_K)
            #     print("rec_df dtypes after adjustment: " + str(rec_df.dtypes))
            if _iteration == 2: #if this is the third run, so we have already evaluated the results with ADJUST_K=50, now we're doing K==0
                rec_df = adjust_for_context(rec_df, propdict, k=0)
                print("rec_df dtypes after adjustment: " + str(rec_df.dtypes))

            # --------- COMBINE RECS AND GROUND TRUTH INTO DF FOR EVALUATION
            if FILTER != "POST":
                logger.info("concatenating ground_truth_df and rec_df into new 'df' for eval")
                df = pd.concat([ground_truth_df, rec_df], ignore_index=True)
            else:
                df = ground_truth_df


            # ---------- Functions to find average ranking of relevant items
            def find_indices(numbers, targets):
                # Create dict to hold indices for each target number. format {item_code: ranking}
                indices = {target: [] for target in targets}

                # Single pass through the list to collect indices
                for index, num in enumerate(numbers):
                    if num in indices:  # Only work with numbers that are targets
                        indices[num].append(index)
                return indices

            def calc_avg_rank_of_relevants(recommended, relevant):
                rankings_of_relevants = find_indices(recommended, relevant)

                values = list(rankings_of_relevants.values())
                if (any(not sublist for sublist in values)): # check for empty rankings
                    print("Something went wrong. There's at least one empty list in rankings_of_relevants.values().")
                    print("Values: ", values)
                    breakpoint()

                return np.mean(values, dtype=np.float32)

            # --------------- PERFORM EVALUATION
            logger.info("starting evaluation")

            def calculate_average_precision(relevant, recommended, k):
                relevant_set = set(relevant)
                score = 0.0
                num_hits = 0.0
                for i, p in enumerate(recommended):
                    if p in relevant_set:
                        num_hits += 1.0
                        score += num_hits / (i + 1.0)
                return score / min(len(relevant), k)

            def calculate_metrics(recommended, relevant):
                if RETRIEVE_ALL_RECS: 
                    avg_rank_of_relevants = calc_avg_rank_of_relevants(recommended, relevant)
                true_positives = len(set(recommended).intersection(set(relevant)))
                
                num_recs = len(recommended)
                num_truths = len(relevant)

                # Calculate precision, recall, and average precision
                precision = true_positives / num_recs if num_recs > 0 else 0
                recall = true_positives / num_truths if num_truths > 0 else 0
                avg_precision = calculate_average_precision(relevant, recommended, len(recommended))
                F1 = (2 * (    ((precision*recall) / (precision+recall))   )) if precision+recall>0 else 0 

                toreturn = {
                    'precision': precision,
                    'recall': recall,
                    'average_precision': avg_precision,
                    'F1': F1
                }
                if RETRIEVE_ALL_RECS: 
                    toreturn['avg_rank_of_relevants'] = avg_rank_of_relevants

                return toreturn

            def get_results(sub_df):
                user_code = sub_df.name

                # ALS
                if FILTER == "POST":
                    ALS_recommended = ALS_recommendations[user_code]
                    adjusted = []
                    for item_code, score in ALS_recommended:
                        adjusted.append((item_code, score * propdict[item_code]))
                    sorted_recs = sorted(adjusted, key=lambda x: x[1], reverse=True)[:TOP_N]
                    ALS_recommended = [item[0] for item in sorted_recs]
                else:
                    ALS_recommended = sub_df.iloc[1]['item_code'][:TOP_N]
                coverage_sets[0].update(ALS_recommended)

                # POPULAR
                if FILTER == "POST":
                    popular_recommended = popular_recs[user_code][:ADJUST_K]
                    adjusted = []
                    for i, item_code in enumerate(popular_recommended):
                        adjusted.append((item_code, i * (1 - propdict[item_code])))
                    sorted_recs = sorted(adjusted, key=lambda x: x[1])[:TOP_N]
                    popular_recommended = [item[0] for item in sorted_recs]
                else:
                    popular_recommended = popular_recs[user_code][:TOP_N]
                coverage_sets[1].update(popular_recommended)

                # RANDOM
                if FILTER == "POST":
                    random_recommended = random.sample(range(num_items_loaded), ADJUST_K)
                    adjusted = []
                    for i, item_code in enumerate(random_recommended):
                        adjusted.append((item_code, i * (1 - propdict[item_code])))
                    sorted_recs = sorted(adjusted, key=lambda x: x[1])[:TOP_N]
                    random_recommended = [item[0] for item in sorted_recs]
                else:
                    random_recommended = random.sample(range(num_items_loaded), TOP_N)
                coverage_sets[2].update(random_recommended)

                # CALCULATE
                relevant = sub_df.iloc[0]['item_code']
                ALS_results.append(calculate_metrics(ALS_recommended, relevant))
                popular_results.append(calculate_metrics(popular_recommended, relevant))
                random_results.append(calculate_metrics(random_recommended, relevant))

            ALS_results = []
            popular_results = []
            random_results = []

            coverage_sets = [set(), set(), set()]

            if FILTER == "POST":
                for key, value in propdict.items():
                    if value < 0.3:
                        propdict[key] = 0.3
                    elif value > 0.7:
                        propdict[key] = 0.7
                    else:
                        propdict[key] = value

            df.groupby('user_code').apply(get_results)

            for i, results in enumerate([ALS_results, popular_results, random_results]):
                model_name = "ALS" if i == 0 else "popular" if i == 1 else "random"
                # breakpoint()
                results = pd.DataFrame(results)
                results = results.describe()

                # ------------- WRITE RESULTS TO FILE
                towrite = []
                if FILTER == "POST":
                    towrite.append("Post filter adjustment applied: " + ADJUSTED)
                towrite.append("test data proportion: " + str(round(len(test_data)/(len(train_data) + len(test_data)), 4)))
                towrite.append("train data length: " + str(len(train_data)))
                towrite.append("test data length: " + str(len(test_data)))
                towrite.append("Items in train: " + (str(interaction_matrix.shape[1]))) # if RECSYS_MODEL=="ALS" else str(train_data["itemId"].nunique())))

                if RETRIEVE_ALL_RECS: 
                    towrite.append("avg rank of relevant items: " + str(results.loc["mean"]["avg_rank_of_relevants"]))

                towrite.append("avg precision: " + str(results.loc["mean"]["precision"]))
                towrite.append("avg recall: " + str(results.loc["mean"]["recall"]))
                towrite.append("avg F1: " + str(results.loc["mean"]["F1"]))
                towrite.append("mean avg precision: " + str(results.loc["mean"]["average_precision"]))
                towrite.append("coverage: " + str(len(coverage_sets[i]) / len(items))) #num_items_loaded))
                
                with open(f"results/{model_name}_{OUTPUT_FILENAME}.txt", "a") as file:
                    file.write("\n")
                    file.write("----- output from run number " + str(run_iterator+1))
                    for result in towrite:
                        file.write("\n")
                        file.write(result)
            logger.critical("evaluation complete, results written to file")

if __name__ == "__main__":
    logger.success("offlineval_all.py was executed directly")

    FILTER = "PRE" # | "PRE" | "POST" | "None"
    CUSTOM_TRAIN_TEST_SPLIT = "is_daytime" # | None | "is_daytime" | "is_weekday" | "TV" | "WEB" | "MOBILE" |
    INVERSE = False # accepted values: True | False | If this is True then is_daytime means is_nighttime
    OUTPUT_FILENAME = "SERIES_test_4weeks"

    TIMES_TO_RUN = 10
    HYPERPARAMS = {'MODEL_PARAM_REGULARIZATION': 0.1, 'MODEL_PARAM_ITERATIONS': 10, 'MODEL_PARAM_FACTORS': 50}
    TOP_N = 10
    TEST_SIZE = 0.25 # 0.210
    NUM_WEEKS = 4
    CONTENT_TYPE_FILTER = "SERIES" # | "None" | "MOVIE" | "SERIES" |

    RETRIEVE_ALL_RECS = False # | True | False |
    ADJUST_K = 50 # put 0 to adjust every item. inactive without FILTER="POST"

    main(FILTER=FILTER, CUSTOM_TRAIN_TEST_SPLIT=CUSTOM_TRAIN_TEST_SPLIT, INVERSE=INVERSE, OUTPUT_FILENAME=OUTPUT_FILENAME, NUM_WEEKS=NUM_WEEKS, TEST_SIZE=TEST_SIZE, \
        CONTENT_TYPE_FILTER=CONTENT_TYPE_FILTER, TOP_N=TOP_N, HYPERPARAMS=HYPERPARAMS, TIMES_TO_RUN=TIMES_TO_RUN, ADJUST_K=ADJUST_K)

