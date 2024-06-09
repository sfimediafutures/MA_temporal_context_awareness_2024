
import pandas as pd
import numpy as np

def parse_contents(contents):
    """
    Returns
        parsed: List of dict
        experiment: dict
        params_to_print: dict
    """
    parsed = []
    experiment = {}
    params_to_print = {}
    for line in contents.split("\n"):
        if line.startswith("----- output from run number"):
            parsed.append(experiment.copy())
        elif line.startswith("avg precision:"):
            avg_precision = line.split(": ")[1]
            experiment["Precision"] = np.float32(avg_precision)

        elif line.startswith("avg recall:"):
            avg_recall = line.split(": ")[1]
            experiment["Recall"] = np.float32(avg_recall)

        elif line.startswith("avg F1:"):
            F1 = line.split(": ")[1]
            experiment["F1"] = np.float32(F1)

        elif line.startswith("mean avg precision:"):
            mean_avg_precision = line.split(": ")[1]
            experiment["MAP"] = np.float32(mean_avg_precision)

        elif line.startswith("coverage:"):
            coverage = line.split(": ")[1]
            experiment["Coverage"] = np.float32(coverage)

        elif line.startswith("test data proportion:"):
            test_data_prop = line.split(": ")[1]
            experiment["Test size"] = np.float32(test_data_prop)

        elif line.startswith("test data length:"):
            test_data_length = line.split(": ")[1]
            experiment["len(test)"] = int(test_data_length)
            
        elif line.startswith("train data length:"):
            train_data_length = line.split(": ")[1]
            experiment["len(train)"] = int(train_data_length)

        elif line.startswith("Items in train:"):
            num_items = line.split(": ")[1]
            experiment["Unique items"] = int(num_items)

        elif line.startswith("avg rank of relevant items:"):
            avg_rank_of_relevants = line.split(": ")[1]
            experiment["Avg rank of relevants"] = np.float32(avg_rank_of_relevants)

        elif line.startswith("Post filter adjustment applied:"):
            experiment["Post-filtered"] = line.split(": ")[1]
#----------------------------------------PARAMS---------------------------------------------------------------
        elif line.startswith("TOP_N:"):
            params_to_print["TOP_N"] = line.split(": ")[1]
        elif line.startswith("ADJUST_K:"):
            params_to_print["ADJUST_K"] = line.split(": ")[1]
        elif line.startswith("NUM_WEEKS:"):
            params_to_print["NUM_WEEKS"] = line.split(": ")[1]
        elif line.startswith("CONTENT_TYPE_FILTER:"):
            params_to_print["CONTENT_TYPE_FILTER"] = line.split(": ")[1]
        elif line.startswith("UNSPLIT_DF_LEN:"):
            params_to_print["UNSPLIT_DF_LEN"] = line.split(": ")[1]
        elif line.startswith("AVG_PROPORTION:"):
            params_to_print["AVG_CONTEXT_PROPORTION"] = round(float(line.split(": ")[1]), 3)

    parsed.append(experiment)
    return parsed[1:], experiment, params_to_print


def overwrite_empty_std(df):
    def overwrite(entry):
        is_number = isinstance(entry, (int, float, np.integer, np.floating))
        if is_number:
            return "" if float(entry) == 0.0 else entry
        return entry

    if (type(df.loc["std",:]) == pd.core.frame.DataFrame):
        df.loc['std'] = df.loc['std'].applymap(overwrite)
    else:
        df.loc["std",:] = df.loc["std",:].map(overwrite)
    return df

# ----------------------FUNCTIONS TO CALL DIRECTLY -------------------------------------------------------------

def fuse(before, after):
    dfs = [before, after]
    most_cols = 0 if (len(before.columns) > len(after.columns)) else 1
    empty_row = pd.DataFrame([[None for _ in range(len(dfs[most_cols].columns))]], columns=dfs[most_cols].columns, index=["DIVIDER"])
    
    if (len(before.columns) - len(after.columns)) == 0:
        result = pd.concat([before, empty_row, after], ignore_index=False)
    else:
        least_cols = dfs[0] if most_cols == 1 else dfs[1]
        cols_to_add = []
        for most_col in list(dfs[most_cols].columns):
            if not most_col in list(least_cols.columns):
                cols_to_add.append(most_col)
        df_to_add = pd.DataFrame([{"mean":"Missing Value", "std":"Missing Value"} for _ in list(least_cols.index)])

        rename_cols={}
        for i in range(len(cols_to_add)):
            rename_cols[i]=cols_to_add[i]
        df_to_add = df_to_add.T.rename(columns=rename_cols)
        
        least_cols = pd.concat([least_cols, df_to_add], axis=1, join="inner")

        if dfs[most_cols].equals(before):
            result = pd.concat([before, empty_row, least_cols], ignore_index=False)
        elif dfs[most_cols].equals(after):
            result = pd.concat([least_cols, empty_row, after], ignore_index=False)
        else:
            print("we have a problem.")

    return result

def finalize_comparison(df, round_floats=True):
    """ Assumes there are these row indices in df: ["mean", "std", "DIVIDER", "mean", "std"] """
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    first_mean = df[numeric_cols].iloc[0]
    second_mean = df[numeric_cols].iloc[3]
    percent_change = ((second_mean - first_mean) / first_mean) * 100

    non_numeric_cols = list(df.columns[~(df.columns).isin(numeric_cols)])
    for col in non_numeric_cols:
        percent_change[col] = ""

    appendant = percent_change.to_frame().rename(columns={"mean":"change in mean"}).T
    if round_floats:
        float_cols = df.select_dtypes(include=['float']).columns
        print(f"these are the float cols: {float_cols}")
        df[float_cols] = df[float_cols].astype(float).round(5)
        appendant[float_cols] = appendant[float_cols].astype(float).round(3)

    def prettify(col):
        colname = col.name  
        if colname in numeric_cols:
            if col.iloc[0] > 0: 
                if colname not in ["Avg rank of relevants", "Percentile relevants"]:
                    col.iloc[0] = "+" + str(col.iloc[0]) + " %"
                else:
                    col.iloc[0] = str(col.iloc[0]*-1) + " %"

            elif col.iloc[0] < 0:
                if colname not in ["Avg rank of relevants", "Percentile relevants"]:
                    col.iloc[0] = str(col.iloc[0]) + " %"
                else:
                    col.iloc[0] = "+" + str(col.iloc[0]*-1) + " %"
            else:
                col.iloc[0] = "NO CHANGE"
        return col

    appendant.apply(prettify)
    if "Test size" in numeric_cols:
        appendant["Test size"] = ""
    if "Unique items" in numeric_cols:
        appendant["Unique items"] = ""

    result= pd.concat([df, appendant], axis=0)
    result.iloc[2, :] = "----"
    return overwrite_empty_std(result)

def post(filename, clean=True, add_folder_prefix=True, return_full_df=False, int_cols_remain_as_int=True):
    if add_folder_prefix: filename = "post/" + filename
    with open(f"{filename}.txt", "r") as file:
        contents = file.read()
        parsed, experiment, params_to_print = parse_contents(contents)
        
    df = pd.DataFrame(parsed)
    keys = experiment.keys()
    if "Avg rank of relevants" in keys:
        df["Percentile relevants"] = df["Avg rank of relevants"] / df["Unique items"] * 100
    print(filename)
    params_print = ""
    for key, value in params_to_print.items():
        params_print += f"{key}: {value}, "

    described = df.describe().loc[["mean", "std"]]
    if "Test size" in keys and "len(train)" in keys and "len(test)" in keys:
        if float(described["Test size"]["std"]) == 0.00:
            params_print += f"len(train): {int(described['len(train)']['mean'])}, "
            params_print += f"len(test): {int(described['len(test)']['mean'])}, "
            df.drop(columns=["len(train)", "len(test)"], inplace=True)

    print(params_print[:-2]) # not printing the final ', ' of the string

    if return_full_df:
        print(df[df["Post-filtered"]=="NO"].describe().loc[["mean", "std"]])
        print("ABOVE was WITHOUT postfiltering. BELOW is WITH postfiltering.")
        print(df[df["Post-filtered"]=="YES"].describe().loc[["mean", "std"]])
        return df
    else:
        if int_cols_remain_as_int:
            int_cols = df.select_dtypes(include=['int']).columns
            # print(int_cols)

        # Calculate descriptive statistics for 'NO'
        stats_no = df[df["Post-filtered"] == "NO"].describe().loc[["mean", "std"]]
        if int_cols_remain_as_int:
            stats_no[int_cols] = stats_no[int_cols].astype("int")
        stats_no["Post-filtered"] = "NO"

        # Create an empty DataFrame with the same structure as stats_no for spacing
        empty_row = pd.DataFrame([[None for _ in range(len(stats_no.columns))]], columns=stats_no.columns, index=["DIVIDER"])

        # Calculate descriptive statistics for 'YES'
        stats_yes = df[df["Post-filtered"] == "YES"].describe().loc[["mean", "std"]]
        if int_cols_remain_as_int:
            stats_yes[int_cols] = stats_yes[int_cols].astype("int")
        stats_yes["Post-filtered"] = "YES"

        # Concatenate the results with an empty row in between
        result = pd.concat([stats_no, empty_row, stats_yes], ignore_index=False)
        float_cols = df.select_dtypes(include=['float']).columns
        result[float_cols] = result[float_cols].round(5)

        if clean:
            result.iloc[2, :] = "----"
            return overwrite_empty_std(result)
        return result

def parse(filename, return_full_df=False, prefix=""):
    filename = prefix + filename
    with open(f"{filename}.txt", "r") as file:
        contents = file.read()
        parsed, experiment, params_to_print = parse_contents(contents)

    df = pd.DataFrame(parsed)
    keys = experiment.keys()
    if "Avg rank of relevants" in keys:
        df["Percentile relevants"] = df["Avg rank of relevants"] / df["Unique items"] * 100
    print(filename)
    params_print = ""
    for key, value in params_to_print.items():
        params_print += f"{key}: {value}, "

    described = df.describe().loc[["mean", "std"]]
    if "Test size" in keys and "len(train)" in keys and "len(test)" in keys:
        if float(described["Test size"]["std"]) == 0.00:
            params_print += f"len(train): {int(described['len(train)']['mean'])}, "
            params_print += f"len(test): {int(described['len(test)']['mean'])}, "
            described.drop(columns=["len(train)", "len(test)"], inplace=True)
    print(params_print[:-2]) # omitting the final 2 characters ', ' of the string

    if return_full_df: 
        print(described)
        return df
    else:
        return described

def pre(filename, overwrite_zero_std=0, return_full_df=False):
    if overwrite_zero_std: return overwrite_empty_std(parse(filename, return_full_df=return_full_df, prefix="pre/"))
    return parse(filename, return_full_df=return_full_df, prefix="pre/")

def base(filename, overwrite_zero_std=0, return_full_df=False):
    if overwrite_zero_std: return overwrite_empty_std(parse(filename, return_full_df=return_full_df, prefix="base/"))
    return parse(filename, return_full_df=return_full_df, prefix="base/")

def simplepost(filename, overwrite_zero_std=0, return_full_df=False):
    if overwrite_zero_std: return overwrite_empty_std(parse(filename, return_full_df=return_full_df, prefix="post/"))
    return parse(filename, return_full_df=return_full_df, prefix="post/")

def oldbase(filename, overwrite_zero_std=0, return_full_df=False):
    if overwrite_zero_std: return overwrite_empty_std(parse("old/base/"+filename, return_full_df=return_full_df))
    return parse("old/base/"+filename, return_full_df=return_full_df)

def oldpre(filename, overwrite_zero_std=0, return_full_df=False):
    if overwrite_zero_std: return overwrite_empty_std(parse("old/pre/"+filename, return_full_df=return_full_df))
    return parse("old/pre/"+filename, return_full_df=return_full_df)

def oldpost(filename, clean=True, return_full_df=False):
    return post("old/post/"+filename, clean=clean, return_full_df=return_full_df, add_folder_prefix=False)

def compare_base_and_pre(base_filename:str, prefiltered_filename:str = ""):
    """uses same filename for both if a second argument is not provided"""
    if prefiltered_filename == "":
        filename = base_filename
        return finalize_comparison(fuse(base(filename), pre(filename)))

    return finalize_comparison(fuse(base(base_filename), pre(prefiltered_filename)))

def old_compare_base_and_pre(base_filename:str, prefiltered_filename:str = ""):
    """uses same filename for both if a second argument is not provided"""
    if prefiltered_filename == "":
        filename = base_filename
        return finalize_comparison(fuse(oldbase(filename), oldpre(filename)))

    return finalize_comparison(fuse(oldbase(base_filename), oldpre(prefiltered_filename)))

def postcompare(filename):
    return finalize_comparison(oldpost(filename, clean=False))

def anycompare(first, second):
    return finalize_comparison(fuse(first, second))

def excel(df, output_filename, experiment_A=False, experiment_B=False):
    
    if experiment_A:
        df = df.reset_index()
        df["index"].iloc[-1] = "in mean"
        df["index"].iloc[2] = "----"

        # Assign new index values for the groups
        group_labels = [experiment_A, experiment_A, "-----", experiment_B, experiment_B, "Difference"]
        df['Group'] = group_labels

        # Create a MultiIndex using 'Group' and the old index
        df.set_index(['Group', 'index'], inplace=True)
        df.index.names = ['Experiment', '']
    df.to_excel(f"excelled/{output_filename}.xlsx", engine="openpyxl")