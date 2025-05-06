import pandas as pd


def check_consistency(df1, df2, id_columns):
    """
    Check if samples with matching values across specified ID columns have identical values 
    across all other columns in both DataFrames.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First DataFrame
    df2 : pandas.DataFrame
        Second DataFrame (subset)
    id_columns : list of str, default=['Id']
        List of column names to use as composite key for matching records
        
    Returns:
    --------
    bool
        True if all matching records are identical, False otherwise
    dict
        Dictionary containing details of any mismatches found
    """
    # Convert single string to list for convenience
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    
    # Validate id_columns exist in both DataFrames
    if not all(col in df1.columns for col in id_columns):
        return False, {"error": f"Not all ID columns {id_columns} found in first DataFrame"}
    if not all(col in df2.columns for col in id_columns):
        return False, {"error": f"Not all ID columns {id_columns} found in second DataFrame"}
    
    # Create composite key for matching
    df1['_composite_key'] = df1[id_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    df2['_composite_key'] = df2[id_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    # Ensure both DataFrames have the same columns
    common_columns = set(df1.columns) & set(df2.columns)
    if not common_columns:
        return False, {"error": "No common columns found between DataFrames"}
    
    # Get common composite keys
    keys_in_both = set(df1['_composite_key']) & set(df2['_composite_key'])
    print(f"keys in both datasets: {len(keys_in_both)}")
    print(f"len of new dataset: {len(df1)}")
    print(f"len of old dataset: {len(old_dataset)}")
    if not keys_in_both:
        return False, {"error": "No matching records found between DataFrames"}
    
    # Filter both DataFrames to only include common keys and columns
    df1_filtered = df1[df1['_composite_key'].isin(keys_in_both)][list(common_columns)]
    df2_filtered = df2[df2['_composite_key'].isin(keys_in_both)][list(common_columns)]
    
    
    # Sort both DataFrames by ID columns to ensure proper comparison
    df1_filtered = df1_filtered.sort_values(id_columns).reset_index(drop=True)
    df2_filtered = df2_filtered.sort_values(id_columns).reset_index(drop=True)
    print(f"len  of new_filtered: {len(df1_filtered)}")
    print(f"len  of old_filtered: {len(df2_filtered)}")
    
    # Compare DataFrames
    is_identical = df1_filtered.equals(df2_filtered)
    
    if is_identical:
        return True, {
            "message": "All matching records are identical",
            "matching_records": len(keys_in_both)
        }
    else:
        # Find and report differences
        differences = {}
        for key in keys_in_both:
            row1 = df1_filtered[df1_filtered['_composite_key'] == key]
            row2 = df2_filtered[df2_filtered['_composite_key'] == key]

            if len(row2) == 0:
                print(f"could not find key: {key} in old dataset")
                print(f"len of row1: {len(row1)}")
                print(f"row1: {row1[id_columns]}")
                print(f"len of row2: {len(row2)}")
                print(f"row2: {row2[id_columns]}")
                print("***********")
                continue

            for col in common_columns:
                if not row1[col].equals(row2[col]):
                    if key not in differences:
                        differences[key] = {
                            'id_values': {id_col: row1[id_col].iloc[0] for id_col in id_columns}
                        }
                    
                    differences[key][col] = {
                        'df1_value': row1[col].iloc[0],
                        'df2_value': row2[col].iloc[0]
                    }
        
        return False, {
            "message": "Differences found in matching records",
            "matching_records": len(keys_in_both),
            "differences": differences
        }
    

def simple_subset_test(super_set: pd.DataFrame, sub_set: pd.DataFrame) -> bool:
    return len(super_set) == len(super_set.merge(sub_set))

def loop_subset_test(super_set: pd.DataFrame, sub_set: pd.DataFrame) -> bool:
    for sub_index, sub_sample in sub_set.iterrows():
        match_found = False
        for super_index, super_sample in super_set.iterrows():
            sample_matches = True
            for col in sub_set.columns.values:
                #print(col)
                #print(sub_sample)
                #print(super_sample)
                if sub_sample[col] != super_sample[col]:
                    sample_matches = False
                    break

            if sample_matches:
                match_found = True
                break 
        if not match_found:
            print(f"could not match sample {sub_sample}")
            return False
    return True

# the purpose of this file is to test that the INTUIT builds samples that are the same a the ones build using the hard coded dictionaries perviously used.

if __name__ == "__main__":
    # load both of the csv files 
    new_dataset_path = "./dataframes/dataset_from_json.csv"
    old_dataset_path = "./dataframes/battery_for_ai.csv"

    with open(new_dataset_path, "r") as new_dataset_file:
        new_dataset = pd.read_csv(new_dataset_file)

    with open(old_dataset_path, "r") as old_dataset_file:
        old_dataset = pd.read_csv(old_dataset_file)

    if set(new_dataset.columns) != set(old_dataset.columns):
        print(f"new_dataset cols: {new_dataset.columns}")
        print(f"old_dataset cols: {old_dataset.columns}")

    print(f"simple subset test: {simple_subset_test(old_dataset, new_dataset)}")
    print(f"loop subset test: {loop_subset_test(old_dataset, new_dataset)}")
    # the columns used to identify individual samples 
    columns_key = ["id",
                "inference_level",
                "pov",
                "spacing_noise_proportion",
                "character_noise_number",
                "character_noise_proportion",
                "capitalisation_number",
                "capitalisation_proportion",
                "condition",
                ]    
    result, differences = check_consistency(new_dataset, old_dataset, columns_key)
    print(f"new_dataset == old dataset: {result}")
    if not result:
        print(f"Differences:\n {differences}")
   

